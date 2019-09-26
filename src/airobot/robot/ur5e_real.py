"""
A UR5e robot with a robotiq 2f140 gripper
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time

import numpy as np
import rospy
from transforms3d.euler import euler2quat, euler2mat

from airobot.robot.robot import Robot
from airobot.sensor.camera.rgbd_cam import RGBDCamera
from airobot.utils import tcp_util
from trac_ik_python import trac_ik
from airobot.utils.common import clamp

class UR5eRobotReal(Robot):
    def __init__(self, cfgs, host, use_cam=False, use_arm=True):
        try:
            rospy.init_node('ur5e', anonymous=True)
        except rospy.exceptions.ROSException:
            rospy.logwarn('ROS node [ur5e] has already been initialized')
        if use_cam:
            self.camera = RGBDCamera(cfgs=cfgs)
        if use_arm:
            super(UR5eRobotReal, self).__init__(cfgs=cfgs)

            self._init_consts()

            self.host = host

            self.monitor = tcp_util.SecondaryMonitor(self.host)
            self.monitor.wait()  # make contact with robot before anything

    def send_program(self, prog):
        """
        Method to send URScript program to the TCP/IP monitor

        Args:
            prog (str): URScript program which will be sent and run on
            the UR5e machine

        Return:
            None
        """
        self.monitor.send_program(prog)

    def output_pendant_msg(self, msg):
        """
        Method to display a text message on the UR5e teach pendant

        Args:
            msg (str): message to display

        Return:
            None
        """
        prog = "textmsg(%s)" % msg
        self.send_program(prog)

    def _is_running(self):
        return self.monitor.running

    def go_home(self):
        """
        Move the robot to a pre-defined home pose
        """
        # 6 joints for the arm, 7th joint for the gripper
        self.set_jpos(self._home_position, wait=True)

    def set_jpos(self, position, joint_name=None, wait=True, *args, **kwargs):
        """
        Method to send a joint position command to the robot

        Args:
            position (float or list): desired joint position(s)
            joint_name (str): If not provided, position should be a list and
                all actuated joints will be moved to specified positions. If
                provided, only specified joint will move
            wait (bool): whether position command should be blocking or non
                blocking

        Return:
            success (bool): whether command was completed successfully or not
        """
        position = copy.deepcopy(position)
        success = False

        if joint_name is None:
            # if len(position) == 6:
            #     gripper_pos = self.get_jpos(self.gripper_jnt_names[0])
            #     position.append(gripper_pos)
            if len(position) != 6:
                raise ValueError("position should contain 6 or 7 elements if"
                                 "joint_name is not provided")

            # gripper_pos = position[-1]
            # gripper_pos = max(self.gripper_close_angle, gripper_pos)
            # gripper_pos = min(self.gripper_open_angle, gripper_pos))

            # position[-1] = gripper_pos

            target_pos = position
        else:
            if joint_name is self.gripper_jnt_names:
                pass  # don't do anything with gripper yet
            else:
                target_pos_joint = position
                current_pos = self.get_jpos()
                rvl_jnt_idx = self.rvl_joint_names.index(joint_name)
                current_pos[rvl_jnt_idx] = target_pos_joint
                target_pos = copy.deepcopy(current_pos)
        prog = "movej([%f, %f, %f, %f, %f, %f])" % (target_pos[0],
                                                    target_pos[1],
                                                    target_pos[2],
                                                    target_pos[3],
                                                    target_pos[4],
                                                    target_pos[5])
        self.send_program(prog)
        if wait:
            success = self._wait_to_reach_jnt_goal(target_pos,
                                                   joint_name=joint_name,
                                                   mode="pos")

        return success

    def set_jvel(self, velocity, acc=0.1, joint_name=None, wait=False,
                 *args, **kwargs):
        """
        Set joint velocity in rad/s

        Args:
            velocity (list): list of target joint velocity values
            joint_name (str, optional): If not provided, velocity should be
                list and all joints will be turned on at specified velocity.
                Defaults to None.
            wait (bool, optional): [description]. Defaults to False.
        """
        velocity = copy.deepcopy(velocity)
        if joint_name is None:
            # if (len(velocity) == 6):
            #     gripper_vel = self.get_jvel(self.gripper_jnt_names[0])
            #     velocity.append(gripper_vel)

            if len(velocity) != 6:
                raise ValueError("Velocity should contain 6 or 7 elements"
                                 "if the joint name is not provided")

            target_vel = velocity

        else:
            if joint_name in self.gripper_jnt_names:
                pass
            else:
                target_vel_joint = velocity
                target_vel = [0.0] * 7
                rvl_jnt_idx = self.rvl_joint_names.index(joint_name)
                target_vel[rvl_jnt_idx] = target_vel_joint

        prog = "speedj([%f, %f, %f, %f, %f, %f], a=%f)" % (target_vel[0],
                                                           target_vel[1],
                                                           target_vel[2],
                                                           target_vel[3],
                                                           target_vel[4],
                                                           target_vel[5],
                                                           acc)
        self.send_program(prog)

        # TODO see about getting back joint velocity info from the robot to
        # use for success flag

    def set_ee_pose(self, pos, ori=None, acc=0.1, vel=0.05, wait=False,
                    use_ik=False, *args, **kwargs):
        """
        Set cartesian space pose of end effector

        Args:
            pos (list): Desired x, y, z positions in the robot's base frame to
                move to
            ori (list, optional): Desired euler angle orientation of the end
                effector. Defaults to None.
            acc (float, optional): Acceleration of end effector during
                beginning of movement. Defaults to 0.1.
            vel (float, optional): Velocity of end effector during movement.
                Defaults to 0.05.

        Returns:
            [type]: [description]
        """
        if ori is None:
            ori = self.get_ee_pose()[-1]  # last index of return is euler angle
        ee_pos = [pos[0], pos[1], pos[2], ori[0], ori[1], ori[2]]
        prog = "movel(p[%f, %f, %f, %f, %f, %f], a=%f, v=%f, r=%f)" % (
            ee_pos[0],
            ee_pos[1],
            ee_pos[2],
            ee_pos[3],
            ee_pos[4],
            ee_pos[5],
            acc,
            vel,
            0.0)
        self.send_program(prog)

        # if wait:
        # TODO implement blocking version

        # TODO implement computeIK version

    def move_ee_xyz(self, delta_xyz, eef_step=0.005, wait=True,
                    *args, **kwargs):
        """Move end effector in straight line while maintaining orientation

        Args:
            delta_xyz (list): Goal change in x, y, z position of end effector
            eef_step (float, optional): [description]. Defaults to 0.005.
        """
        # success = True
        ee_pos, ee_quat, ee_rot_mat, ee_euler = self.get_ee_pose()

        # current_pos = np.array(ee_pos)
        # delta_xyz = np.array(delta_xyz)
        # path_len = np.linalg.norm(delta_xyz)
        # num_pts = int(np.ceil(path_len / float(eef_step)))
        # if num_pts <= 1:
        #     num_pts = 2

        # waypoints_sp = np.linspace(0, path_len, num_pts).reshape(-1, 1)
        # waypoints = current_pos + waypoints_sp / float(path_len) * delta_xyz

        # for i in range(waypoints.shape[0]):
        #     self.set_ee_pose(waypoints[i, :].flatten().tolist(), ee_quat)
        #     time.sleep(0.01)

        # or instead of this ^ just
        ee_pos[0] += delta_xyz[0]
        ee_pos[1] += delta_xyz[1]
        ee_pos[2] += delta_xyz[2]

        self.set_ee_pose(ee_pos, ee_euler, wait=wait)

    def get_jpos(self, joint_name=None, wait=False):
        """Get current joint angles of robot

        Args:
            joint_name (str, optional): Defaults to None.

        Return:
            jpos (list): list of current joint positions in radians
        """
        jdata = self.monitor.get_joint_data(wait)
        jpos = [jdata["q_actual0"], jdata["q_actual1"], jdata["q_actual2"],
                jdata["q_actual3"], jdata["q_actual4"], jdata["q_actual5"]]
        return jpos

    def get_jvel(self, joint_name=None, wait=False):
        """Get current joint angular velocities of robot

        Args:
            joint_name (str, optional): Defaults to None.

        Return:
            jvel (list): list of current joint angular velocities in radians/s
        """
        jdata = self.monitor.get_joint_data(wait)
        jvel = [jdata["qd_actual0"], jdata["qd_actual1"], jdata["qd_actual2"],
                jdata["qd_actual3"], jdata["qd_actual4"], jdata["qd_actual5"]]
        return jvel

    def get_ee_pose(self, wait=False):
        """Get current cartesian pose of the EE, in the robot's base frame

        TODO: why do we need to wait here?

        Args:
            wait (bool, optional): [description]. Defaults to False.

        Returns:
            list: x, y, z position of the EE
            list: quaternion representation of the EE orientation
            list: rotation matrix representation of the EE orientation
            list: euler angle representation of the EE orientation (roll, pitch, yaw with
                static reference frame)
        """
        pose_data = self.monitor.get_cartesian_info(wait)
        if pose_data:
            pos = [pose_data["X"], pose_data["Y"], pose_data["Z"]]
            euler_ori = [pose_data["Rx"], pose_data["Ry"], pose_data["Rz"]]

            rot_mat = euler2mat(euler_ori[0],
                                euler_ori[1],
                                euler_ori[2]).flatten().tolist()
            quat_ori = euler2quat(euler_ori[0],
                                  euler_ori[1],
                                  euler_ori[2]).flatten().tolist()

        return pos, quat_ori, rot_mat, euler_ori

    def compute_ik(self, ee_pose):
        """
        Function to obtain the joint angles corresponding
        to a particular end effector pose

        Args:
            ee_pose (list): End effector pose, specified as
                [x, y, z, roll, pitch, yaw]
        """
        raise NotImplementedError

    def _wait_to_reach_jnt_goal(self, goal, joint_name=None, mode='pos'):
        """
        Block the code to wait for the joint moving to the specified goal.
        The goal can be a desired velocity(s) or a desired position(s).
        Max waiting time is self.cfgs.TIMEOUT_LIMIT

        Args:
            goal (float or list): goal positions or velocities
            joint_name (str): if it's none, all the actuated
                joints are compared.
                Otherwise, only the specified joint is compared
            mode (str): 'pos' or 'vel'

        Returns:
            if the goal is reached or not
        """
        success = False
        start_time = time.time()
        while True:
            if not self._is_running():
                raise RuntimeError("Robot stopped")

            if time.time() - start_time > self.cfgs.TIMEOUT_LIMIT:
                pt_str = 'Unable to move to joint goals [mode: %s] (%s)' \
                         ' within %f s' % (mode, str(goal),
                                           self.cfgs.TIMEOUT_LIMIT)
                arutil.print_red(pt_str)
                return success
            if self._reach_jnt_goal(goal, joint_name, mode=mode):
                success = True
                break
            time.sleep(0.001)
        return success

    def _reach_jnt_goal(self, goal, joint_name=None, mode='pos'):
        """
        Check if the joint reached the goal or not.
        The goal can be a desired velocity(s) or a desired position(s).

        Args:
            goal (float or list): goal positions or velocities
            joint_name (str): if it's none, all the
                actuated joints are compared.
                Otherwise, only the specified joint is compared
            mode (str): 'pose' or 'vel'

        Returns:
            if the goal is reached or not
        """
        goal = np.array(goal)
        if mode == 'pos':
            new_jnt_val = self.get_jpos(joint_name)
        elif mode == 'vel':
            new_jnt_val = self.get_jvel(joint_name)
        else:
            raise ValueError('Only pos and vel modes are supported!')
        new_jnt_val = np.array(new_jnt_val)
        jnt_diff = new_jnt_val - goal
        error = np.max(np.abs(jnt_diff))
        if error < self.cfgs.MAX_JOINT_ERROR:
            return True
        else:
            return False

    def _init_consts(self):
        """
        Initialize constants
        """
        self._home_position = [1.57, -1.5, 2.0, -2.05, -1.57, 0]

        self.arm_jnt_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        self.arm_jnt_names_set = set(self.arm_jnt_names)
        self.arm_dof = len(self.arm_jnt_names)
        self.gripper_jnt_names = [
            'finger_joint', 'left_inner_knuckle_joint',
            'left_inner_finger_joint', 'right_outer_knuckle_joint',
            'right_inner_knuckle_joint', 'right_inner_finger_joint'
        ]
        self.gripper_close_angle = 0.7
        self.gripper_open_angle = 0
        self.gripper_jnt_names_set = set(self.gripper_jnt_names)
        self.rvl_joint_names = self.arm_jnt_names + self.gripper_jnt_names
        self._ik_jds = [self._ik_jd] * len(self.rvl_joint_names)
        self.ee_link = 'wrist_3_link-tool0_fixed_joint'

        # https://www.universal-robots.com/how-tos-and-faqs/faq/ur-faq/max-joint-torques-17260/
        self._max_torques = [150, 150, 150, 28, 28, 28]
        # a random value for robotiq joints
        self._max_torques.append(20)
        # self.camera = PyBulletCamera(p, self.cfgs)

        robot_description = self.cfgs.ROBOT_DESCRIPTION
        urdf_string = rospy.get_param(robot_description)
        self.num_ik_solver = trac_ik.IK(self.cfgs.ROBOT_BASE_FRAME,
                                        self.cfgs.ROBOT_EE_FRAME,
                                        urdf_string=urdf_string)

    def scale_moveit_motion(self, vel_scale=1.0, acc_scale=1.0):
        vel_scale = clamp(vel_scale, 0.0, 1.0)
        acc_scale = clamp(acc_scale, 0.0, 1.0)
        self.moveit_group.set_max_velocity_scaling_factor(vel_scale)
        self.moveit_group.set_max_acceleration_scaling_factor(acc_scale)

    def _close(self):
        self.monitor.close()
