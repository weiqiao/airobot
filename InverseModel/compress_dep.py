import os
import cv2
import numpy as np

scaling = 10000.0
data_dirs = ['image_85_depth', 'image_86_depth', 'image_87_depth', 'image_88_depth']

for write_dir in data_dirs:
    load_path = os.path.join('../data/', write_dir)
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    for dep_name in os.listdir(load_path):
        dep = np.load(os.path.join(load_path, dep_name))
        sdep = dep * scaling
        cv2.imwrite(write_dir + '/' + dep_name[:-3] + 'png', sdep.astype(np.uint16))
