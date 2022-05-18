import random
import os, sys
# from os.path import *
# import tkinter as tk
import cv2

from FUSION.script.image_management import name_generator
from Stereo_matching.Tools.disparity_tools import reprojection_disparity

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
# import lynred_py
# import cv2
# from cv2 import waitKey
# from imutils.video import VideoStream
# from numpy import linspace
# from math import tan, pi
# from FUSION.classes.Camera import Camera
# from FUSION.tools import gradient_tools
# from FUSION.tools.data_management_tools import register_cmap
# from FUSION.tools.manipulation_tools import *
# # from FUSION.tools.method_fusion import colormap_fusion
# from FUSION.tools.gradient_tools import *
# from FUSION.tools.mapping_tools import generate_support_map
from FUSION.tools.method_fusion import *
from FUSION.tools.registration_tools import *
# import numpy as np
# from FUSION.interface.Application import Application
# import time
# from scipy.ndimage import median_filter
#
# from Stereo_matching.Tools.disparity_tools import reprojection_disparity



# with open("/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/hybrid/Calibration/transform_matrix", "rb") as f:
#     disparity = np.array(pickle.load(f))
# print(disparity)
Time = 'Day'
number = name_generator(random.randint(0, 99))
im_type = 'visible'

im_infrared_aligned = ImageCustom("/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/hybrid/infrared_projected/left"+ number + ".png")
imgR = ImageCustom("/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/hybrid/right/right" + number + ".png").BGR()
# lab = imgR.LAB()
# lab[lab[:, :, 0] > 250, 0] = im_infrared_aligned[lab[:, :, 0] > 250]
# imgR = lab.BGR()
b, g, r = imgR[:, :, 0], imgR[:, :, 1], imgR[:, :, 2]
b = b#/2 + r / 4 + g / 4
r = 3*r / 4 + im_infrared_aligned / 4
g = im_infrared_aligned / 2 + g/2
imgL = np.stack([b, g, r], axis=2)/255



# fusion = lab.BGR()

cv.imshow('Modified Infrared image', imgL)
cv.imshow('Infrared image', im_infrared_aligned)
cv.imshow('Color image', imgR)
# fus = (imgR*0.5 + imgL*0.5)/255
# cv.imshow('new_image', imgL)depth
# cv.imshow('image', disparity/m)
cv.waitKey(0)

with open("/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/hybrid/new_disparity/disp" + number, "rb") as f:
    new_disparity = np.array(pickle.load(f))/2
depth = 100 / (new_disparity + 0.005)
depth[depth > 50] = 50
depth = 1 - depth/50
