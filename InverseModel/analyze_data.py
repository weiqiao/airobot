import os,sys
import os.path 
from os import path, listdir
from os.path import isfile, join
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import atexit
import time
import datetime
import pickle
import signal
import torch
import torchvision
from torch import nn
from torch.utils import data
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, models
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import linecache
from PIL import Image
from matplotlib.image import imread
from torch.optim.lr_scheduler import StepLR

# des_path = 'rel_dist2_err_dir/rel_dist_err_1_100_'
# des_path = 'rel_dist2_err_dir_render/rel_dist_err_1_100_'
# des_path = 'rel_dist3_2_err_dir/rel_dist_err_1_100_'
des_path = 'rel_dist3_err_dir/rel_dist_err_1_100_'
dat = np.zeros((1,6))
for index in range(1,101):
	d = np.load(des_path + str(index) + '.npy')
	if d[-1] > 1:
		continue
	l = len(d)
	if l < 6:
		x = d[-1]
		while len(d) < 6:
			d = np.append(d, x)
	dat = np.concatenate((dat, d.reshape(1,-1)))
des_path = 'rel_dist3_2_err_dir/rel_dist_err_1_100_'
for index in range(1,101):
	d = np.load(des_path + str(index) + '.npy')
	if d[-1] > 1:
		continue
	l = len(d)
	if l < 6:
		x = d[-1]
		while len(d) < 6:
			d = np.append(d, x)
	dat = np.concatenate((dat, d.reshape(1,-1)))
	# print(d)
	# dat = np.concatenate((dat, d.reshape(1,-1)), axis=0)
dat = np.delete(dat, 0, 0)
print(dat)
l = len(dat)
dat = np.concatenate((np.ones((l, 1)), dat), axis=1)
dat_mean = np.mean(dat, axis=0)
dat_std = np.std(dat, axis=0)
print(dat_mean, dat_std)
# from IPython import embed
# embed()
print(len(dat))
plt.errorbar(range(7), dat_mean, yerr=dat_std, linewidth=3, fmt='-', color='r')
# plt.plot(range(7), dat_mean)
plt.show()
