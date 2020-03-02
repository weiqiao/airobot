import os
import cv2
import numpy as np
import pybullet as p
from environment import PokingEnv


class ReplayPoke(PokingEnv):
    def __init__(self):
        super(ReplayPoke, self).__init__(True)
        self.pokes = [[0, 0, 0, 0]] # zero poke as the starting poke
        self.posis = []
        self.quats = []
        pos, quat, _, _ = self.get_box_pose()
        self.posis.append(pos)
        self.quats.append(quat)


    def replay(self, save_dir='data/temp'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        poke_index = 0
        while True:
            # generate random poke through box center
            obj_x, obj_y = self.posis[poke_index][0:2]
            poke_ang, poke_len, start_x, start_y, end_x, end_y = self.sample_poke(obj_x, obj_y)
            self.execute_poke(start_x, start_y, end_x, end_y)
            self.pokes.append([start_x, start_y, end_x, end_y])
            poke_index += 1

            # log new box pos
            pos, quat, _, _ = self.get_box_pose()
            self.posis.append(pos)
            self.quats.append(quat)

            # set start box
            self.box_id2 = self.load_box(self.posis[poke_index-1],
                                        self.quats[poke_index-1],
                                        [0, 1, 0, 0.5])
            p.setCollisionFilterPair(self.box_id2, self.box_id, -1, -1, enableCollision=0)

            # # set goal box
            # self.box_id3 = self.load_box(self.posis[poke_index],
            #                              self.quats[poke_index],
            #                              [0, 0, 1, 0.5])
            # p.setCollisionFilterPair(self.box_id3, self.box_id2, -1, -1, enableCollision=0)
            # p.setCollisionFilterPair(self.box_id3, self.box_id, -1, -1, enableCollision=0)
            # p.setCollisionFilterPair(self.box_id3, 1, -1, 10, enableCollision=0)
            rgb1, _ = self.get_img()

            # do poke
            stx, sty, edx, edy = self.pokes[poke_index]
            self.execute_poke(stx, sty, edx, edy)
            rgb2, _ = self.get_img()

            # remove boxes
            self.remove_box(self.box_id2)
            # self.remove_box(self.box_id3)

            # log images
            cv2.imwrite(save_dir + '/' + str(poke_index*2-2) + '.png',
                        cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_dir + '/' + str(poke_index*2-1) + '.png',
                        cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR))
