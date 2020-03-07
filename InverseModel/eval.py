import pybullet as p
from environment import *


# label index in the poke data array (31, 1)
# index 0 is the poke index corresponding to starting image
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
    def __init__(self, ifRender=False):
        self.env = PokingEnv(ifRender)
        self.gt = None # ground_truth
        self.gt_file = None # ground_truth file name
        self.pd = None # prediction
        self.box_id2 = None # goal box id


    def load_pokedata(self, gtfile, pdfile=None):
        """
        loading ground_truth and predictions
        """
        self.gt = np.loadtxt(gtfile)
        self.gt_file = gtfile
        if pdfile is not None:
            self.pd = np.loadtxt(pdfile)


    def validate_data(self, num_to_see=None, tag='world'):
        """
        validate that data is consistent: obj_start pose + poke -> obj_end pose
        visualize ground truth pokes and predicted pokes
        tag: world, wpoke, joint, pixel, default None correspond to ground truth
        """
        if self.gt is None:
            raise ValueError('load ground truth')
        if self.box_id2 is not None:
            self.env.remove_box(self.box_id2)
        if num_to_see is None:
            query = range(len(self.gt) - 1) # query the starting obj pose
        else:
            query = np.random.randint(0, len(self.gt)-1, num_to_see)
        for i in query:
            poke = self.gt[i, stx:edy+1]
            obj_start = self.gt[i, obx:qt4+1]
            obj_end = self.gt[i+1, obx:qt4+1]
            if tag == 'world': # ground truth pokes in world frame
                pass
            elif tag == 'wpoke': # gt pokes calc from poke angle and length
                poke_ang, poke_len = self.gt[i, pang], self.gt[i, plen]
                poke = self.wpoke2world(obj_start[0], obj_start[1], poke_ang, poke_len)
            elif tag == 'pixel': # gt pokes calc from pixels and depth images
                rows = np.array([self.gt[i, sr], self.gt[i, edr]])
                cols = np.array([self.gt[i, stc], self.gt[i, edc]])
                if self.gt_file is None:
                    raise ValueError('load ground truth')
                depth_file = self.gt_file[:-4] + '_depth/' + str(i) + '.npy'
                poke = self.pixel2world(rows, cols, depth_file)
            elif tag == 'joint': # gt pokes calc from joint values
                pass
            else:
                raise ValueError('tag has to be world, wpoke, pixel or joint')

            _ = self.eval_poke(poke, obj_start, obj_end)


    def eval_poke(self, poke, obj_start, obj_end):
        """
        evaluate/visualize a single poke
        obj_start and obj_end: (x, y, q1, q2, q3, q4)
        """
        self.env.reset_box(pos=[obj_start[0], obj_start[1], self.env.box_z],
                      quat=obj_start[-4:])
        self.set_goal(pos=[obj_end[0], obj_end[1], self.env.box_z],
                      quat=obj_end[-4:])
        init_dist = np.sqrt((obj_start[0]-obj_end[0])**2 + (obj_start[1]-obj_end[1])**2)
        _, _ = self.env.execute_poke(poke[0], poke[1], poke[2], poke[3])
        _, _ = self.env.get_img()
        curr_pos, _, _, _ = self.env.get_box_pose()
        dist = np.sqrt((curr_pos[0]-obj_end[0])**2 + (curr_pos[1]-obj_end[1])**2)
        relative_location_error = dist/init_dist
        # remove target box
        self.env.remove_box(self.box_id2)
        self.box_id2 = None
        print('%.3f %.3f' % (dist, relative_location_error))
        return relative_location_error


    def forward_poke(self, poke_dir, attempt_idx):
        """
        use poke in 'attempt_idx'.txt to forward simulate the effects on obj_pose in
        'attempt_idx-1'.txt
        """
        for i in range(len(os.listdir(poke_dir))):
            poke_path = os.path.join(poke_dir, str(i).zfill(2))
            # set box pose
            prev_state = np.loadtxt(os.path.join(poke_path, str(attempt_idx-1)+'.txt'))
            box_pose = prev_state[1]
            box_target = prev_state[2]
            self.env.go_home()
            self.env.reset_box(pos=box_pose[:3], quat=box_pose[3:])
            self.set_goal(box_target[:3], box_target[3:])
            # poke
            poke = np.loadtxt(os.path.join(poke_path, str(attempt_idx)+'.txt'))
            if len(poke) == 3:
                poke = poke[0]
            _, _ = self.env.execute_poke(poke[0], poke[1], poke[2], poke[3])
            after_rgb, after_dep = self.env.get_img()
            after_img = self.env.resize_rgb(after_rgb)
            cv2.imwrite(poke_path + '/' + str(attempt_idx) + '.png', # 5 refers to goal state
                        cv2.cvtColor(after_img, cv2.COLOR_RGB2BGR))
            np.save(poke_path + '/' + str(attempt_idx), after_dep)
            after_pos, after_quat, _, _ = self.env.get_box_pose()
            self.env.remove_box(self.box_id2)
            # log poke
            with open(os.path.join(poke_path, str(attempt_idx)+'.txt'), 'w') as file:
                file.write('%f %f %f %f %f %f %f\n' % \
                     (poke[0], poke[1], poke[2], poke[3], 0, 0, 0)) # 0s are placeholders
                file.write('%f %f %f %f %f %f %f\n' % \
                     (after_pos[0], after_pos[1], after_pos[2],
                      after_quat[0], after_quat[1], after_quat[2], after_quat[3]))
                file.write('%f %f %f %f %f %f %f\n' % \
                     (box_target[0], box_target[1], box_target[2],
                      box_target[0], box_target[1], box_target[2], box_target[3]))




    def set_goal(self, pos, quat):
        self.box_id2 = self.env.load_box(pos=pos, quat=quat, rgba=[0,0, 1,0.5])
        p.setCollisionFilterPair(self.env.box_id, self.box_id2, -1, -1, enableCollision=0)
        p.setCollisionFilterPair(self.box_id2, 1, -1, 10, enableCollision=0) # with arm
        # _, _ = self.env.get_img()


    def batch_eval(self, gtfile, pdfile, tag, test_num=1):
        """
        batch eval predictions
        """
        # pred data comes in [indices, pred_pokes, true_pokes]
        # wpoke indexs
        wpoke_ang, wpoke_len = 1, 2
        wpoke_ang_gt, wpoke_len_gt = 3, 4
        # world indexs
        world_stx, world_sty, world_edx, world_edy = 1, 2, 3, 4
        world_stx_gt, world_sty_gt, world_edx_gt, world_edy_gt = 7, 8, 9, 10
        # pixel indexs
        pixel_str, pixel_stc, pixel_edr, pixel_edc = 1, 2, 3, 4
        pixel_str_gt, pixel_stc_gt, pixel_edr_gt, pixel_edc_gt = 7, 8, 9, 10
        # joint indexs
        self.load_pokedata(gtfile, pdfile)
        if (self.gt is None) or (self.pd is None):
            raise ValueError('load ground truth and prediction file')
        accu_dist = 0.0
        # query = np.random.randint(0, len(self.pd)-1, test_num) # len-1 avoids end row access
        query = range(100)
        dist_list = []
        for i in query:
            img_idx = int(self.pd[i][0])
            obj_start = self.gt[img_idx, obx:qt4+1]
            obj_end = self.gt[img_idx+1, obx:qt4+1]
            if tag == 'gt':
                # batch eval ground_truth pokes
                poke = self.gt[img_idx, stx:edy+1]
            elif tag == 'world':
                poke = self.pd[i][world_stx:world_edy+1]
            elif tag == 'wpoke':
                poke_ang = self.pd[i][wpoke_ang]
                poke_len = self.pd[i][wpoke_len]
                print('ang: ', poke_ang, ' len: ', poke_len)
                poke = self.wpoke2world(obj_start[0], obj_start[1], poke_ang, poke_len)
            elif tag == 'pixel':
                rows = np.array([self.pd[i, pixel_str], self.pd[i, pixel_edr]])
                cols = np.array([self.pd[i, pixel_stc], self.pd[i, pixel_edc]])
                depth_file = gtfile[:-4] + '_depth/' + str(i) + '.npy'
                poke = self.pixel2world(rows, cols, depth_file)
            else:
                raise ValueError('tag has to be gt, wpoke, world, joint and pixel')
            curr_dist = self.eval_poke(poke, obj_start, obj_end)
            accu_dist += curr_dist
            dist_list.append(curr_dist)
        np.savetxt('poke_eval/relative_location_error.txt', np.array(dist_list))
        return accu_dist / len(query)


    def wpoke2world(self, obj_x, obj_y, poke_ang, poke_len):
        """
        take obj position, poke angle and poke length to return poke in world frame
        """
        start_x = obj_x - self.env.poke_len_min * np.cos(poke_ang)
        start_y = obj_y - self.env.poke_len_min * np.sin(poke_ang)
        end_x = obj_x + poke_len * np.cos(poke_ang)
        end_y = obj_y + poke_len * np.sin(poke_ang)
        return [start_x, start_y, end_x, end_y]


    def pixel2world(self, rows, cols, depth_file):
        """
        given a depth image, and rows and cols in image space,
        predict x and y poking location in world frame
        """
        rows = ((rows*self.env.row_scale + self.env.row_min) + 0.5).astype(int)
        cols = ((cols*self.env.col_scale) + 0.5).astype(int)
        depth_im = np.load(depth_file)
        pokes_3d = self.env.pixel2xyz(depth_im, rows, cols)
        # print(pokes_3d)
        poke = pokes_3d[:, :2].flatten()
        return poke


    def joint2world(self):
        pass





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='enables rendering')
    parser.add_argument('--gt', default=None, help='gt file')
    parser.add_argument('--pd', default=None, help='pred file')
    parser.add_argument('--tag', default='gt', help='world, wpoke, joint or pixel, default is gt')
    parser.add_argument('--num', type=int, default=100, help='test num')
    args = parser.parse_args()
    eva = EvalPoke(ifRender=True)

    if (args.gt is not None) and (args.pd is not None):
        accu_dist = eva.batch_eval(args.gt, args.pd, args.tag, args.num)
        print('')
        print('accu_dist for ', args.pd)
        print('tag: ', args.tag)
        print('%.3f' % accu_dist)
        print('')











#
