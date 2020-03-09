import os,sys
import os.path 
from os import path, listdir
from os.path import isfile, join

import cv2
import argparse
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.image import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils, models
from torch.optim.lr_scheduler import StepLR

from airobot import Robot, log_info
from airobot.utils.common import euler2quat, quat2euler
import pybullet as p
from learn_poke_dynamics import *

class PokingEnv(object):
	"""
	Poking environment: loads initial object on a table
	"""
	def __init__(self, ifRender=False):
		#np.set_printoptions(precision=3, suppress=True)
		# table scaling and table center location
		self.table_scaling = 0.6 # tabletop x ~ (0.3, 0.9); y ~ (-0.45, 0.45)
		self.table_x = 0.6
		self.table_y = 0
		self.table_z = 0.6
		self.table_surface_height = 0.975 # get from running robot.cam.get_pix.3dpt
		self.table_ori = euler2quat([0, 0, np.pi/2])
		# task space x ~ (0.4, 0.8); y ~ (-0.3, 0.3)
		self.max_arm_reach = 0.91
		self.workspace_max_x = 0.75 # 0.8 discouraged, as box can move past max arm reach
		self.workspace_min_x = 0.4
		self.workspace_max_y = 0.3
		self.workspace_min_y = -0.3
		# robot end-effector
		self.ee_min_height = 0.99
		self.ee_rest_height = 1.1 # stick scale="0.0001 0.0001 0.0007"
		self.ee_home = [self.table_x, self.table_y, self.ee_rest_height]
		# initial object location
		self.box_z = 1 - 0.005
		self.box_pos = [self.table_x, self.table_y, self.box_z]
		self.box_size = 0.02 # distance between center frame and size, box size 0.04
		# poke config: poke_len by default [0.06-0.1]
		self.poke_len_min = 0.06 # 0.06 ensures no contact with box empiracally
		self.poke_len_range = 0.04
		# image processing config
		self.row_min = 40
		self.row_max = 360
		self.col_min = 0
		self.col_max = 640
		self.output_row = 100
		self.output_col = 200
		self.row_scale = (self.row_max - self.row_min) / float(self.output_row)
		self.col_scale = (self.col_max - self.col_min) / float(self.output_col)
		assert self.col_scale == self.row_scale
		# load robot
		self.robot = Robot('ur5e_stick', pb=True, pb_cfg={'gui': ifRender})
		self.robot.arm.go_home()
		self.ee_origin = self.robot.arm.get_ee_pose()
		self.go_home()
		self._home_jpos = self.robot.arm.get_jpos()
		# load table
		self.table_id = self.load_table()
		# load box
		self.box_id = self.load_box()
		# initialize camera matrices
		self.robot.cam.setup_camera(focus_pt=[0.7, 0, 1.],
		                            dist=0.5, yaw=90, pitch=-60, roll=0)
		self.ext_mat = self.robot.cam.get_cam_ext()
		self.int_mat = self.robot.cam.get_cam_int()

	def go_home(self):
		self.set_ee_pose(self.ee_home, self.ee_origin[1])

	def set_ee_pose(self, pos, ori=None, ignore_physics=False):
		jpos = self.robot.arm.compute_ik(pos, ori)
		return self.robot.arm.set_jpos(jpos, wait=True, ignore_physics=ignore_physics)
    
	def load_table(self):
		return self.robot.pb_client.load_urdf('table/table.urdf',
				[self.table_x, self.table_y, self.table_z],
				self.table_ori,
				scaling=self.table_scaling)


	def load_box(self, pos=None, quat=None, rgba=[1, 0, 0, 1]):
		if pos is None:
			pos = self.box_pos
		return self.robot.pb_client.load_geom('box', size=self.box_size,
				mass=1,
				base_pos=pos,
				base_ori=quat,
				rgba=rgba)


	def reset_box(self, box_id=None, pos=None, quat=None):
		if box_id is None:
			box_id = self.box_id
		if pos is None:
			pos = self.box_pos
		return self.robot.pb_client.reset_body(box_id, pos, quat)

	def move_ee_xyz(self, delta_xyz):
		return self.robot.arm.move_ee_xyz(delta_xyz, eef_step=0.015)

	def execute_poke(self, start_x, start_y, end_x, end_y, i=None):
		cur_ee_pose = self.robot.arm.get_ee_pose()[0]
		self.move_ee_xyz([start_x-cur_ee_pose[0], start_y-cur_ee_pose[1], 0])
		self.move_ee_xyz([0, 0, self.ee_min_height-self.ee_rest_height])
		self.move_ee_xyz([end_x-start_x, end_y-start_y, 0]) # poke
		print('ok')
		self.move_ee_xyz([0, 0, self.ee_rest_height-self.ee_min_height])
		# move arm away from camera view
		self.go_home() # important to have one set_ee_pose every loop to reset accu errors
		new_ee_pose = self.robot.arm.get_ee_pose()[0]
		return new_ee_pose

	def get_box_pose(self, box_id=None):
		if box_id is None:
		    box_id = self.box_id
		pos, quat, lin_vel, _ = self.robot.pb_client.get_body_state(box_id)
		rpy = quat2euler(quat=quat)
		return pos, quat, rpy, lin_vel

	def get_img(self):
		rgb, depth = self.robot.cam.get_images(get_rgb=True, get_depth=True)
		# crop the rgb
		img = rgb[40:360, 0:640]
		# dep = depth[40:360, 0:640]
		# low pass filter : Gaussian blur
		# blurred_img = cv2.GaussianBlur(img.astype('float32'), (5, 5), 0)
		small_img = cv2.resize(img.astype('float32'), dsize=(200, 100),
		            interpolation=cv2.INTER_CUBIC) # numpy array dtype numpy int64 by default
		# small_dep = cv2.resize(dep.astype('float32'), dsize=(200, 100),
		#             interpolation=cv2.INTER_CUBIC) # numpy array dtype numpy int64 by default

		# small_img = cv2.cvtColor(small_img,cv2.COLOR_RGB2BGR)
		return small_img, depth

obx, oby, qt1, qt2, qt3, qt4 = 5, 6, 7, 8, 9, 10 # obj pose before poke

class EvalPoke():
	"""
	Calculating poking statistics for trained models
	"""
	def __init__(self, 
		ifRender=False,
		index=1,
		model_path = 'learn_poke_dynamics_data/exp_2020-03-01-23-02-26/model',
		model_number='model_5.pth'):
		self.env = PokingEnv(ifRender)
		self.gt = None # ground_truth
		self.gt_file = None # ground_truth file name
		self.pd = None # prediction
		self.box_id2 = None # goal box id
		self.jnt_dim = 4

		self.model = InverseModel(200,200,self.jnt_dim).float().cuda()
		self.model.load_state_dict(torch.load( os.path.join(model_path, model_path) ))

		self.transform = transforms.Compose([transforms.ToTensor(),
							transforms.Normalize(mean=[0.5, 0.5, 0.5],
							std=[0.5, 0.5, 0.5])])

		self.des_path = '280kpokes/image_88'
		self.index = index
		self.initial_img = imread( os.path.join(self.des_path, str(self.index-1) + '.png') )
		self.final_img = imread( os.path.join(self.des_path, str(self.index) + '.png') )
		self.data = np.loadtxt(self.des_path + '.txt')
		self.obj_start = self.data[self.index-1, obx:qt4+1]
		self.obj_end = self.data[self.index, obx:qt4+1]

	def eval_poke(self):
		self.env.reset_box(pos=[self.obj_start[0], self.obj_start[1], self.env.box_z],
						quat=self.obj_start[-4:])
		i = 0
		# self.env.move_ee_xyz([0, 0, self.env.ee_min_height-self.env.ee_rest_height])

		# import pdb; pdb.set_trace()
		# self.set_goal(pos=[self.obj_end[0], self.obj_end[1], self.env.box_z],
		# 				quat=self.obj_end[-4:])
		while True:
			curr_pos, _, _, _ = self.env.get_box_pose()
			dist = np.sqrt((curr_pos[0]-self.obj_end[0])**2 + (curr_pos[1]-self.obj_end[1])**2)
			cost = dist
			print(cost)
			if cost < 0.02:
				break

			small_img, depth = self.env.get_img()
			cv2.imwrite('norender' + str(i) + '.png', small_img)
			small_img /= 255.
			# import pdb; pdb.set_trace()
			## compute action
			# img = np.vstack([self.initial_img, self.final_img])
			img = np.vstack([small_img, self.final_img])
			# print(img)
			img = self.transform(img).reshape(1,3,200,200).cuda()
			# print(img.detach().cpu().numpy())
			act = self.model(img)
			act = act.detach().cpu().numpy()[0]
			# print(act)

			## execute poke 
			cur_ee_pose = self.env.execute_poke(act[0], act[1], act[2], act[3], i)

			i += 1
			if i > 5:
				break 

		print(cost)

	def set_goal(self, pos, quat):
		self.box_id2 = self.env.load_box(pos=pos, quat=quat, rgba=[0,0, 1,0.5])
		p.setCollisionFilterPair(self.env.box_id, self.box_id2, -1, -1, enableCollision=0)
		p.setCollisionFilterPair(self.box_id2, 1, -1, 10, enableCollision=0) # with arm


if __name__ == '__main__':
	a = EvalPoke(index=6)
	a.eval_poke()