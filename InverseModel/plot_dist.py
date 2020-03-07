import os,sys
import os.path 
from os import path, listdir
from os.path import isfile, join
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


a = np.array([0.08985145969877174, 0.03243872650160903, 0.07111077081813355,0.0257969772724227,0.07852030964671117,0.08576537145196626,0.08576537145196626])

b = np.array([0.09414555229536868,0.08076574688213035,0.052535867048992525,0.053123743281534866,0.05312527674798453,0.05312553096257893,0.05312553096257893])

c = np.array([0.08943969688007673,0.02664188254111474,0.08876241420034171,0.03137303996078268,0.08112677972226005,0.04229740491228155,0.04229740491228155])
plt.plot(np.asarray(range(len(a))), a)
plt.plot(np.asarray(range(len(b))), b)
plt.plot(np.asarray(range(len(c))), c)
# plt.title('training/validation loss')
# plt.legend(['training inverse loss', 'validation inverse loss'],fontsize='xx-large')
plt.show()
