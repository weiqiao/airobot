import os
import argparse
import time
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import airobot as ar
from airobot import Robot
from airobot.utils.common import euler2quat
from airobot.utils.common import quat2euler
from airobot.utils.pb_util import load_urdf
from airobot.utils.pb_util import load_geom
from airobot.utils.pb_util import get_body_state
from airobot.utils.pb_util import reset_body
import pybullet as p
from airobot.utils.pb_util import remove_body

# under scale factor
table_length = 1.0
table_wideth = 0.6 # but effective arm reach is 0.9
table_scaling = 0.6
arm_span = 0.9
min_height = 1.1 #+ 0.02
rest_height = 1.2 # stick scale="0.0001 0.0001 0.0007"
home = [0.6, 0, rest_height]
table_x = 0.59 # task space x ~ (0.4, 0.75); y ~ (-0.3, 0.3)
table_y = 0
table_z = 0.60
box_z = 1
box_pos = [table_x, table_y, box_z]
box_size = 0.02
workspace_max_x = 0.8 - 0.05
workspace_min_x = 0.4
workspace_max_y = 0.4 - 0.1
workspace_min_y = -0.4 + 0.1
poke_len_mean=0.06 # poke_len by default [0.06-0.1]
poke_len_std=0.04


def main(ifRender=False):
    """
    this function loads initial oboject on a table,
    """
    np.set_printoptions(precision=3, suppress=True)


    # load robot
    robot = Robot('ur5e', use_eetool=False, arm_cfg={'render': ifRender, 'self_collision': True})
    robot.arm.go_home()
    origin = robot.arm.get_ee_pose()

    def go_home():
        robot.arm.set_ee_pose(home, origin[1])
        # robot.arm.eetool.close()
    # init robot
    go_home()

    # load table
    table_ori = euler2quat([0, 0, np.pi/2.0])
    table_pos = [table_x, table_y, table_z]
    table_id = load_urdf('table/table.urdf', table_pos, table_ori, scaling=table_scaling)

    # load box
    box_id = load_geom('box', size=box_size, mass=1, base_pos=box_pos, rgba=[1, 0, 0, 1])

    # init camera
    def get_img():
        # screenshot camera images
        focus_pt = [0.7, 0, 1.]
        robot.cam.setup_camera(focus_pt=focus_pt, dist=0.5, yaw=90, pitch=-60, roll=0)
        rgb, depth = robot.cam.get_images(get_rgb=True, get_depth=True)
        # crop the rgb
        img = rgb[40:360, 0:640]
        # dep = depth[40:360, 0:640]
        # low pass filter : Gaussian blur
        # blurred_img = cv2.GaussianBlur(img.astype('float32'), (5, 5), 0)
        small_img = cv2.resize(img.astype('float32'), dsize=(200, 100),
                    interpolation=cv2.INTER_CUBIC) # numpy array dtype numpy int64 by default
        # small_dep = cv2.resize(dep.astype('float32'), dsize=(200, 100),
        #             interpolation=cv2.INTER_CUBIC) # numpy array dtype numpy int64 by default
        return small_img, depth

    # test run
    def test():
        time_to_sleep = 0.5
        go_home()
        pose = robot.arm.get_ee_pose()
        robot.arm.set_ee_pose([pose[0][0], pose[0][1], min_height], origin[1])
        time.sleep(time_to_sleep)
        get_img()
        # test boundary
        robot.arm.set_ee_pose([pose[0][0], pose[0][1]+table_length/2.0, min_height], origin[1])
        # robot.arm.eetool.close()
        time.sleep(time_to_sleep)
        get_img()
        robot.arm.move_ee_xyz([0, -table_length, 0])
        time.sleep(time_to_sleep)
        get_img()
        pose = robot.arm.get_ee_pose()
        robot.arm.set_ee_pose([pose[0][0], pose[0][1], rest_height], origin[1])
        time.sleep(time_to_sleep)
        get_img()
        robot.arm.move_ee_xyz([0, table_length, 0])
        time.sleep(time_to_sleep)
        get_img()
        pose = robot.arm.get_ee_pose()
        robot.arm.set_ee_pose([pose[0][0], pose[0][1], min_height], origin[1])
        time.sleep(time_to_sleep)
        get_img()
        # test arc
        for i in list([np.pi/3.0, np.pi*2/5.0, np.pi*3/7.0, np.pi/2.0,
                       np.pi*4/7.0, np.pi*3/5.0, np.pi*2/3.0]):
            robot.arm.set_ee_pose([arm_span*np.sin(i),
            arm_span*np.cos(i), rest_height], origin[1])
            time.sleep(time_to_sleep)
            get_img()
            robot.arm.set_ee_pose([arm_span*np.sin(i),
            arm_span*np.cos(i), min_height], origin[1])
            time.sleep(time_to_sleep)
            get_img()
            robot.arm.set_ee_pose([arm_span*np.sin(i),
            arm_span*np.cos(i), rest_height], origin[1])
            time.sleep(time_to_sleep)
            get_img()

    def get_pixel(X, Y, Z=0.975):
        """return fractional pixels representations
           Z values comes from running robot.cam.get_pix.3dpt
           representing the table surface height"""
        ext_mat = robot.cam.get_cam_ext()
        int_mat = robot.cam.get_cam_int()
        XYZ = np.array([X, Y, Z, 1])
        xyz = np.linalg.inv(ext_mat).dot(XYZ)[:3]
        pixel_xyz = int_mat.dot(xyz)
        pixel_xyz = pixel_xyz / pixel_xyz[2]
        pixel_col = pixel_xyz[0] / 3.2 # due to image cropping and scaling
        pixel_row = (pixel_xyz[1] - 40) / 3.2
        return pixel_row, pixel_col

    # # log object
    # pos, quat, lin_vel, ang_vel = get_body_state(box_id)
    # ar.log_info('Box:')
    # ar.log_info('     position: %s' % np.array2string(pos, precision=2))
    # ar.log_info('     quaternion: %s' % np.array2string(quat, precision=2))
    # ar.log_info('     linear vel: %s' % np.array2string(lin_vel, precision=2))
    # ar.log_info('     angular vel: %s' % np.array2string(ang_vel, precision=2))

    # plt.figure()
    # plt.imshow(rgb)
    # plt.figure()
    # plt.imshow(depth * 25, cmap='gray', vmin=0, vmax=255)
    # print('Maximum Depth (m): %f' % np.max(depth))
    # print('Minimum Depth (m): %f' % np.min(depth))
    # plt.show()

    def poke(save_dir='data/image'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_depth_dir = save_dir + '_depth'
        if not os.path.exists(save_depth_dir):
            os.makedirs(save_depth_dir)
        index = 0
        curr_img, curr_dep = get_img()
        while True:
            obj_pos, obj_quat, lin_vel, _ = get_body_state(box_id)
            obj_ang = quat2euler(obj_quat)[2] # -pi ~ pi
            obj_x, obj_y, obj_z = obj_pos
            # check if cube is on table
            if obj_z < box_z / 2.0 or lin_vel[0] > 1e-3: # important that box is still
                print(obj_z, lin_vel[0])
                reset_body(box_id, box_pos)
                continue
            while True:
                # choose random poke point on the object
                poke_x = np.random.random()*box_size + obj_x - box_size/2.0
                poke_y = np.random.random()*box_size + obj_y - box_size/2.0
                # choose poke angle along the z axis
                poke_ang = np.random.random() * np.pi * 2 - np.pi
                # choose poke length
                poke_len = np.random.random() * poke_len_std + poke_len_mean
                # calc starting poke location and ending poke loaction
                start_x = poke_x - poke_len * np.cos(poke_ang)
                start_y = poke_y - poke_len * np.sin(poke_ang)
                end_x = poke_x + poke_len * np.cos(poke_ang)
                end_y = poke_y + poke_len * np.sin(poke_ang)
                if end_x > workspace_min_x and end_x < workspace_max_x \
                    and end_y > workspace_min_y and end_y < workspace_max_y:
                    break
            robot.arm.move_ee_xyz([start_x-home[0], start_y-home[1], 0], 0.015)
            robot.arm.move_ee_xyz([0, 0, min_height-rest_height], 0.015)
            js1, js2, js3, js4, js5, js6 = robot.arm.get_jpos()
            robot.arm.move_ee_xyz([end_x-start_x, end_y-start_y, 0], 0.015)
            je1, je2, je3, je4, je5, je6 = robot.arm.get_jpos()
            # important that we use move_ee_xyz, as set_ee_pose can throw obj in motion
            robot.arm.move_ee_xyz([0, 0, rest_height-min_height], 0.015)
            # move arm away from camera view
            go_home() # important to have one set_ee_pose every loop to reset accu errors
            next_img, next_dep = get_img()
            # calc poke and obj locations in image pixel space
            start_r, start_c = get_pixel(start_x, start_y)
            end_r, end_c = get_pixel(end_x, end_y)
            obj_r, obj_c = get_pixel(obj_x, obj_y)
            with open(save_dir + '.txt', 'a') as file:
                file.write('%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % \
                    (index, start_x, start_y, end_x, end_y, obj_x, obj_y,
                    obj_quat[0], obj_quat[1], obj_quat[2], obj_quat[3],
                    js1, js2, js3, js4, js5, js6, je1, je2, je3, je4, je5, je6,
                    start_r, start_c, end_r, end_c, obj_r, obj_c))
            cv2.imwrite(save_dir + '/' + str(index) +'.png',
                        cv2.cvtColor(curr_img, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(save_dir + '_depth/' + str(index) +'.png', curr_dep)
            np.save(save_dir + '_depth/' + str(index), curr_dep)
            curr_img = next_img
            curr_dep = next_dep
            if index % 1000 == 0:
                ar.log_info('number of pokes: %sk' % str(index/1000))
            index += 1

    def test_data(filename):
        data = np.loadtxt(filename, unpack=True)
        idx, stx, sty, edx, edy, obx, oby, qt1, qt2, qt3, qt4, js1, js2, js3, js4, js5, js6, je1, je2, je3, je4, je5, je6 = data
        go_home()
        reset_body(box_id, box_pos)
        _, _ = get_img()
        for i in range(5):
            robot.arm.set_ee_pose([stx[i], sty[i], rest_height], origin[1])
            robot.arm.move_ee_xyz([0, 0, min_height-rest_height], 0.015)
            robot.arm.move_ee_xyz([edx[i]-stx[i], edy[i]-sty[i], 0], 0.015)
            robot.arm.move_ee_xyz([0, 0, rest_height-min_height], 0.015)
            _, _ = get_img()
            start_jpos = robot.arm.compute_ik([stx[i], sty[i], min_height])
            end_jpos = robot.arm.compute_ik([edx[i], edy[i], min_height])
            print('start')
            print(js1[i], js2[i], js3[i], js4[i], js5[i], js6[i])
            print(start_jpos)
            print('end')
            print(je1[i], je2[i], je3[i], je4[i], je5[i], je6[i])
            print(end_jpos)
            time.sleep(1)

    def eval_poke(tag, ground_truth, img_idx, poke):
        # ground_truth is the entire matrix from 'image_xx.txt' file
        # img_idx is image index, first column of ground_truth
        # poke is the predicted 4 vector for world, 8 for pixel and joint
        # second half of the 8 vector are world coordinates computed from gt pixel and joint values

        # label index in the poke data array (29, 1)
        # index 0 is the poke index corresponding to starting image
        stx, sty, edx, edy = 1, 2, 3, 4 # ee pos of start and end poke
        obx, oby, qt1, qt2, qt3, qt4 = 5, 6, 7, 8, 9, 10 # obj pose before poke
        js1, js2, js3, js4, js5, js6 = 11, 12, 13, 14, 15, 16 # jpos before poke
        je1, je2, je3, je4, je5, je6 = 17, 18, 19, 20, 21, 22 # jpos after poke
        sr, stc, edr, edc, obr, obc = 23, 24, 25, 26, 27, 28 # row and col locations in image

        gt = ground_truth[img_idx:img_idx+2]
        tgt_posi = gt[1, 5:8]
        tgt_posi[-1] = box_z
        tgt_quat = gt[1, 7:11]
        init_posi = gt[0, 5:8]
        init_posi[-1] = box_z
        init_quat = gt[0, 7:11]
        gt_poke = gt[0, 1:5]
        go_home()
        reset_body(box_id, init_posi, init_quat)

        box_id2 = load_geom('box', size=box_size, mass=1,
            base_pos=tgt_posi, base_ori=tgt_quat, rgba=[0,0,1,0.5])
        p.setCollisionFilterPair(box_id, box_id2, -1, -1, enableCollision=0)
        # to check robot_id and link_id
        # robot.arm.robot_id
        # robot.arm.p.getNumJoints(robot_id)
        # robot.arm.p.getJointInfo(robot_id, 0-max_lnik_id)
        p.setCollisionFilterPair(box_id2, 1, -1, 11, enableCollision=0)
        _, _ = get_img()
        # time.sleep(1)
        # reset_body(box_id, init_posi, init_quat)
        # robot.arm.move_ee_xyz([gt_poke[0]-home[0], gt_poke[1]-home[1], 0], 0.015)
        # robot.arm.move_ee_xyz([0, 0, min_height-rest_height], 0.015)
        # robot.arm.move_ee_xyz([gt_poke[2]-gt_poke[0], gt_poke[3]-gt_poke[1], 0], 0.015)
        # robot.arm.move_ee_xyz([0, 0, rest_height-min_height], 0.015)
        # go_home()
        # _, _ = get_img()
        # reset_body(box_id, init_posi, init_quat)
        # poke = ground_truth[img_idx, 1:5]
        robot.arm.move_ee_xyz([poke[0]-home[0], poke[1]-home[1], 0], 0.015)
        robot.arm.move_ee_xyz([0, 0, min_height-rest_height], 0.015)
        robot.arm.move_ee_xyz([poke[2]-poke[0], poke[3]-poke[1], 0], 0.015)
        robot.arm.move_ee_xyz([0, 0, rest_height-min_height], 0.015)
        go_home()
        _, _ = get_img()
        curr_posi, _, _, _ = get_body_state(box_id)
        dist = np.sqrt((tgt_posi[0]-curr_posi[0])**2 + (tgt_posi[1]-curr_posi[1])**2)
        print(dist)
        # time.sleep(0.5)
        remove_body(box_id2)
        return dist

    def calc_dist(gtfile, pdfile, tag, test_num=50):
        # this function generate 50 visualization sequence
        true = np.loadtxt(gtfile)
        pred = np.loadtxt(pdfile)
        accu_dist = 0.0
        # query = random.sample(range(0, pred.shape[0]), test_num)
        total = pred.shape[0]
        query = range(total)[::total/test_num]
        for i in query:
            img_idx = int(pred[i][0])
            if tag == 'world':
                poke = pred[i][1:5]
            elif tag == 'pixel':
                rows = np.array([pred[i, 1], pred[i, 3], pred[i, 7], pred[i, 9]])
                cols = np.array([pred[i, 2], pred[i, 4], pred[i, 8], pred[i, 10]])
                rows = ((rows*3.2 + 40) + 0.5).astype(int)
                cols = ((cols*3.2) + 0.5).astype(int)
                depth_file = gtfile[:-4] + '_depth/' + str(img_idx) + '.npy'
                depth_im = np.load(depth_file)
                pokes_3d = get_pix_3dpt(depth_im, rows, cols)
                poke = pokes_3d[:, :2].flatten()
            else:
                raise Exception("experiment_tag has to be 'world', 'joint', or 'pixel'")

            accu_dist += eval_poke(tag, true, img_idx, poke)
        return accu_dist / len(query)


    def get_pix_3dpt(depth_im, rs, cs, in_world=True, filter_depth=False,
                     k=1, ktype='median', depth_min=None, depth_max=None):

        if not isinstance(rs, int) and not isinstance(rs, list) and \
                not isinstance(rs, np.ndarray):
            raise TypeError('rs should be an int, a list or a numpy array')
        if not isinstance(cs, int) and not isinstance(cs, list) and \
                not isinstance(cs, np.ndarray):
            raise TypeError('cs should be an int, a list or a numpy array')
        if isinstance(rs, int):
            rs = [rs]
        if isinstance(cs, int):
            cs = [cs]
        if isinstance(rs, np.ndarray):
            rs = rs.flatten()
        if isinstance(cs, np.ndarray):
            cs = cs.flatten()
        if not (isinstance(k, int) and (k % 2) == 1):
            raise TypeError('k should be a positive odd integer.')
        # _, depth_im = self.get_images(get_rgb=False, get_depth=True)
        if k == 1:
            depth_im = depth_im[rs, cs]
        else:
            depth_im_list = []
            if ktype == 'min':
                ktype_func = np.min
            elif ktype == 'max':
                ktype_func = np.max
            elif ktype == 'median':
                ktype_func = np.median
            elif ktype == 'mean':
                ktype_func = np.mean
            else:
                raise TypeError('Unsupported ktype:[%s]' % ktype)
            for r, c in zip(rs, cs):
                s = k // 2
                rmin = max(0, r - s)
                rmax = min(self.img_height, r + s + 1)
                cmin = max(0, c - s)
                cmax = min(self.img_width, c + s + 1)
                depth_im_list.append(ktype_func(depth_im[rmin:rmax,
                                                cmin:cmax]))
            depth_im = np.array(depth_im_list)

        depth = depth_im.reshape(-1) * 1 #self.depth_scale
        img_pixs = np.stack((rs, cs)).reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
        depth_min = depth_min if depth_min else 0.01 #self.depth_min
        depth_max = depth_max if depth_max else 10 #self.depth_max
        if filter_depth:
            valid = depth > depth_min
            valid = np.logical_and(valid,
                                   depth < depth_max)
            depth = depth[:, valid]
            img_pixs = img_pixs[:, valid]

        get_img() # to set up camera matrices
        cam_int_mat_inv = robot.cam.cam_int_mat_inv
        cam_ext_mat = robot.cam.cam_ext_mat
        uv_one = np.concatenate((img_pixs,
                                 np.ones((1, img_pixs.shape[1]))))
        uv_one_in_cam = np.dot(cam_int_mat_inv, uv_one)
        pts_in_cam = np.multiply(uv_one_in_cam, depth)
        if in_world:
            if cam_ext_mat is None:
                raise ValueError('Please call set_cam_ext() first to set up'
                                 ' the camera extrinsic matrix')
            pts_in_cam = np.concatenate((pts_in_cam,
                                         np.ones((1, pts_in_cam.shape[1]))),
                                        axis=0)
            pts_in_world = np.dot(cam_ext_mat, pts_in_cam)
            pts_in_world = pts_in_world[:3, :].T
            return pts_in_world
        else:
            return pts_in_cam.T


    from IPython import embed
    embed()

    # time.sleep(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # omit argument to produce no rendering
    parser.add_argument('--render', type=bool, help='if rendering')
    args = parser.parse_args()
    main(ifRender=args.render)
