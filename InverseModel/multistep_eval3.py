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
from learn_poke_dynamics3 import *


# test multi-step for learn_poke_dynamics3.py
class PokingEnv3(object):
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

		self.stick_size = 0.015

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

	def remove_box(self, box_id):
		self.robot.pb_client.remove_body(box_id)

stx, sty, edx, edy = 1, 2, 3, 4 # ee pos of start and end poke
obx, oby, qt1, qt2, qt3, qt4 = 5, 6, 7, 8, 9, 10 # obj pose before poke
js1, js2, js3, js4, js5, js6 = 11, 12, 13, 14, 15, 16 # jpos before poke
je1, je2, je3, je4, je5, je6 = 17, 18, 19, 20, 21, 22 # jpos after poke
sr, stc, edr, edc, obr, obc = 23, 24, 25, 26, 27, 28 # row and col locations in image
pang, plen = 29, 30 # poke angle and poke length

class EvalPoke():
	"""
	Calculating poking statistics for trained models
	"""
	def __init__(self, 
		ifRender=False,
		index=1,
		box_pos = None,
		model_path = 'learn_poke_dynamics_data/exp_2020-03-01-23-02-26/model',
		model_number='model_5.pth'):
		self.env = PokingEnv3(ifRender)
		self.gt = None # ground_truth
		self.gt_file = None # ground_truth file name
		self.pd = None # prediction
		self.box_id2 = None # goal box id
		self.jnt_dim = 4

		feature_dim = 100
		self.model = InverseModel(200,200,200,feature_dim,200,200,self.jnt_dim).float().cuda()
		self.model.load_state_dict(torch.load( os.path.join(model_path, model_number) ))

		self.transform = transforms.Compose([transforms.ToTensor(),
							transforms.Normalize(mean=[0.5, 0.5, 0.5],
							std=[0.5, 0.5, 0.5])])

		self.box_pos = box_pos # box_pos computed from no render
		self.index = index
		self.des_path = '280kpokes/image_88'
		self.data = np.loadtxt(self.des_path + '.txt')
		self.obj_start = self.data[self.index-1, obx:qt4+1]
		self.obj_end = self.data[self.index, obx:qt4+1]
		self.obj_end[0] += 0.2
		self.obj_end[1] += 0.2

		if not ifRender:
			self.initial_img = imread( os.path.join(self.des_path, str(self.index-1) + '.png') )
			self.final_img = self.image_generator() #imread( os.path.join(self.des_path, str(self.index) + '.png') )
			self.final_img_transformed = self.transform(self.final_img).reshape(1,3,100,200).cuda()

	def image_generator(self):
		env = PokingEnv3()
		env.reset_box(pos=[self.obj_end[0], self.obj_end[1], env.box_z],
                      quat=self.obj_end[-4:])
		img, _ = env.get_img()
		return img


	def eval_poke(self):
		self.env.reset_box(pos=[self.obj_start[0], self.obj_start[1], self.env.box_z],
						quat=self.obj_start[-4:])
		init_cost = 0
		act_list = []
		rel_dist_err = []
		box_pos = np.zeros((1,7))
		for i in range(6):
			curr_pos, curr_quat, _, _ = self.env.get_box_pose()
			box_pos = np.concatenate((box_pos, np.concatenate((curr_pos, curr_quat)).reshape(1,-1)), axis=0)
			dist = np.sqrt((curr_pos[0]-self.obj_end[0])**2 + (curr_pos[1]-self.obj_end[1])**2)
			cost = dist
			if dist < 0.04:
				break
			if i == 0:
				init_cost = cost
				# print(cost/init_cost)
			else:	
				rel_dist_err.append(cost/init_cost)

			small_img, depth = self.env.get_img()
			cv2.imwrite('norender' + str(i) + '.png', small_img)
			small_img /= 255.
			small_img = self.transform(small_img).reshape(1,3,100,200).cuda()
			act = self.model.inverse(small_img, self.final_img_transformed)
			### see how lorge the forward model output is 
			# from IPython import embed
			# embed()
			# phi_img = self.model.encoder(small_img)
			# self.model.forward_dynamics(torch.cat((phi_img, act),1))
			act = act.detach().cpu().numpy()[0]
			l = self.env.box_size + self.env.stick_size
			if abs(act[0] - curr_pos[0]) < l:
				if act[0] < curr_pos[0]:
					act[0] = curr_pos[0] - l
				else:
					act[0] = curr_pos[0] + l
			if abs(act[1] - curr_pos[1]) < l:
				if act[1] < curr_pos[1]:
					act[1] = curr_pos[1] - l
				else:
					act[1] = curr_pos[1] + l
			act_list.append(act)
			## execute poke 
			cur_ee_pose = self.env.execute_poke(act[0], act[1], act[2], act[3], i)

		curr_pos, curr_quat, _, _ = self.env.get_box_pose()
		# save box pos
		box_pos = np.concatenate((box_pos, np.concatenate((curr_pos, curr_quat)).reshape(1,-1)), axis=0)
		box_pos = np.delete(box_pos, 0, 0)
		# save cost
		dist = np.sqrt((curr_pos[0]-self.obj_end[0])**2 + (curr_pos[1]-self.obj_end[1])**2)
		cost = dist
		rel_dist_err.append(cost/init_cost)

		return act_list, rel_dist_err, box_pos 

	def set_goal(self, pos, quat):
		self.box_id2 = self.env.load_box(pos=pos, quat=quat, rgba=[0,0, 1,0.5])
		p.setCollisionFilterPair(self.env.box_id, self.box_id2, -1, -1, enableCollision=0)
		p.setCollisionFilterPair(self.box_id2, 1, -1, 10, enableCollision=0) # with arm

	def animation(self,act_list):
		self.env.reset_box(pos=[self.obj_start[0], self.obj_start[1], self.env.box_z],
						quat=self.obj_start[-4:])
		init_cost = 1
		rel_dist_err = []
		l1 = len(act_list)
		for i in range(l1):
			if self.box_pos is not None:
				self.env.reset_box(pos=[self.box_pos[i,0], self.box_pos[i,1], self.env.box_z],
						quat=self.box_pos[i,-4:])
			self.set_goal(pos=[self.obj_end[0], self.obj_end[1], self.env.box_z],
						quat=self.obj_end[-4:])
			# from IPython import embed
			# embed()
			act = act_list[i]
			curr_pos = self.env.get_box_pose()[0]
			dist = np.sqrt((curr_pos[0]-self.obj_end[0])**2 + (curr_pos[1]-self.obj_end[1])**2)
			if i == 0:
				init_cost = dist
			else:
				rel_dist_err.append(dist/init_cost)
				print(dist/init_cost)
			# check collision between object and stick
			l = self.env.box_size + self.env.stick_size
			if abs(act[0] - curr_pos[0]) < l:
				if act[0] < curr_pos[0]:
					act[0] = curr_pos[0] - l
				else:
					act[0] = curr_pos[0] + l
			if abs(act[1] - curr_pos[1]) < l:
				if act[1] < curr_pos[1]:
					act[1] = curr_pos[1] - l
				else:
					act[1] = curr_pos[1] + l
			# from IPython import embed
			# embed()
			cur_ee_pose = self.env.execute_poke(act[0], act[1], act[2], act[3], i)
			self.env.remove_box(self.box_id2)

		curr_pos = self.env.get_box_pose()[0]
		if self.box_pos is not None:
			self.env.reset_box(pos=[self.box_pos[-1,0], self.box_pos[-1,1], self.env.box_z],
					quat=self.box_pos[-1,-4:])
		dist = np.sqrt((curr_pos[0]-self.obj_end[0])**2 + (curr_pos[1]-self.obj_end[1])**2)
		cost = dist
		rel_dist_err.append(cost/init_cost)
		time.sleep(4)
		return rel_dist_err


class EvalOneStepPoke():
	def __init__(self, 
		ifRender=False,
		index=1,
		model_path = 'learn_poke_dynamics_data/exp_2020-03-01-23-02-26/model',
		model_number='model_5.pth'):
		self.env = PokingEnv3(ifRender)
		self.gt = None # ground_truth
		self.gt_file = None # ground_truth file name
		self.pd = None # prediction
		self.box_id2 = None # goal box id
		self.jnt_dim = 4
		feature_dim = 100
		self.model = InverseModel(200,200,200,feature_dim,200,200,self.jnt_dim).float().cuda()
		self.model.load_state_dict(torch.load( os.path.join(model_path, model_number) ))
		self.transform = transforms.Compose([transforms.ToTensor(),
							transforms.Normalize(mean=[0.5, 0.5, 0.5],
							std=[0.5, 0.5, 0.5])])
		self.des_path = '280kpokes/image_88'
		self.index = index
		self.initial_img = imread( os.path.join(self.des_path, str(self.index-1) + '.png') )
		self.final_img = imread( os.path.join(self.des_path, str(self.index) + '.png') )
		self.final_img_transformed = self.transform(self.final_img).reshape(1,3,100,200).cuda()
		self.data = np.loadtxt(self.des_path + '.txt')
		self.obj_start = self.data[self.index-1, obx:qt4+1]
		self.obj_end = self.data[self.index, obx:qt4+1]

	def test_one_step_online(self):
		self.env.reset_box(pos=[self.obj_start[0], self.obj_start[1], self.env.box_z],
						quat=self.obj_start[-4:])
		self.set_goal(pos=[self.obj_end[0], self.obj_end[1], self.env.box_z],
					quat=self.obj_end[-4:])
		curr_pos = self.env.get_box_pose()[0]
		init_dist = np.sqrt((curr_pos[0]-self.obj_end[0])**2 + (curr_pos[1]-self.obj_end[1])**2)
		# check collision between object and stick

		# compute action
		small_img, depth = self.env.get_img()
		small_img /= 255.
		small_img = self.transform(small_img).reshape(1,3,100,200).cuda()
		act = self.model.inverse(small_img, self.final_img_transformed)
		act = act.detach().cpu().numpy()[0]

		cur_ee_pose = self.env.execute_poke(act[0], act[1], act[2], act[3])
		self.env.remove_box(self.box_id2)
		curr_pos = self.env.get_box_pose()[0]
		dist = np.sqrt((curr_pos[0]-self.obj_end[0])**2 + (curr_pos[1]-self.obj_end[1])**2)
		rel_dist_err = dist/init_dist
		return rel_dist_err

	def set_goal(self, pos, quat):
		self.box_id2 = self.env.load_box(pos=pos, quat=quat, rgba=[0,0, 1,0.5])
		p.setCollisionFilterPair(self.env.box_id, self.box_id2, -1, -1, enableCollision=0)
		p.setCollisionFilterPair(self.box_id2, 1, -1, 10, enableCollision=0) # with arm

if __name__ == '__main__':
	args = sys.argv[1:]
	parser = argparse.ArgumentParser()
	parser.add_argument('--index', type=int)
	parser.add_argument('--stage', type=int, default=0)
	args = parser.parse_args()
	index=args.index
	model_path='learn_poke_dynamics3_data/exp_2020-03-05-00-06-16'
	model_number='model_2.pth'
	# model_number='model_5.pth'

	# a = EvalOneStepPoke(index=index, model_path=model_path, model_number=model_number)
	# rel_dist_err = a.test_one_step_online()
	# np.save('rel_dist3_err_dir_no_render_onestep/rel_dist_err_1_100_' + str(index), rel_dist_err)

	# individual trials
	if args.stage == 0:
		a = EvalPoke(index=index,
			model_path=model_path,
			model_number=model_number)
		act_list,rel_dist_err,box_pos = a.eval_poke()
		print(rel_dist_err)
		np.save('act_list3_dir/act_list_1_100_' + str(index), act_list)
		np.save('rel_dist3_err_dir/rel_dist_err_1_100_' + str(index), rel_dist_err)
		np.save('box_pos3_dir/box_pos_1_100_' + str(index), box_pos)
		print(act_list)	
	else:	
		act_list = np.load('act_list3_dir/act_list_1_100_' + str(index) + '.npy')
		box_pos = np.load('box_pos3_dir/box_pos_1_100_' + str(index) + '.npy')
		# box_pos = None
		# from IPython import embed
		# embed()
		b = EvalPoke(index=index,
			ifRender = True,
			box_pos = box_pos,
			model_path=model_path,
			model_number=model_number)
		rel_dist_err = b.animation(act_list)
		np.save('rel_dist3_err_dir_render/rel_dist_err_1_100_' + str(index), rel_dist_err)

