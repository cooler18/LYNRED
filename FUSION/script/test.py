# import random
# import os
# from os.path import *
# import tkinter as tk
#
# import lynred_py
# from cv2 import waitKey
# from imutils.video import VideoStream
# from numpy import linspace
# from math import tan, pi
# from FUSION.classes.Camera import Camera
# from FUSION.tools import gradient_tools
# from FUSION.tools.data_management_tools import register_cmap
# from FUSION.tools.manipulation_tools import *
# from FUSION.tools.method_fusion import colormap_fusion
# from FUSION.tools.gradient_tools import *
# from FUSION.tools.mapping_tools import generate_support_map
# from FUSION.tools.method_fusion import *
from ..tools.registration_tools import *
# import numpy as np
# from FUSION.interface.Application import Application
# import time
# from scipy.ndimage import median_filter
from Stereo_matching.Algorithms.SGM.OpenCv_DepthMap.depthMapping import depthMapping

#
# url = "http://azorgz:tuorpmoi@192.168.1.150:8080/video"
# # url = "http://azorgz:tuorpmoi@10.124.255.8:8080/video"
#
# app = Camera(tk.Tk(), url)
# app.mainloop()

if __name__ == '__main__':
    imageL, imageR, maps, m, M = depthMapping()
    reconstruction_from_disparity(imageL, imageR, maps, m, M, orientation=0, verbose=True)

# p = join("D:\Travail\LYNRED\FUSION", "Images_grouped")
# path = p + "/visible"
# pathgray = p + "/infrared"
# pathfus = p + "/multispectral"
# #
# n = random.randint(0, len(os.listdir(path)) - 1)
# imageRGB_name = path + "/VIS_" + str(n) + ".jpg"
# imageIR_name = pathgray + "/IFR_" + str(n) + ".tiff"
# imageFUS_name = pathfus + "/MUL_" + str(n) + ".jpg"
#
# imageRGB = ImageCustom(imageRGB_name)
# imageIR = ImageCustom(imageIR_name)
#
#
# maps = generate_support_map((240, 320), 0.5, min_slide=-14, max_slide=23)
# plt.matshow(maps)
# plt.show()
# Wavelet_pyr(imageIR, level=1)
# Wavelet_pyr(imageRGB.GRAYSCALE(), level=2)
# k = 3
# gray_blur = cv.bilateralFilter(imageIR, k, k * 2, k / 2)  # To perserve edges

# 3x3 sobel filters for edge detection
# sobel_x = np.array([[-1, 0, 1],
#                     [-2, 0, 2],
#                     [-1, 0, 1]])
# sobel_y = np.array([[-1, -2, -1],
#                     [0, 0, 0],
#                     [1, 2, 1]])
#
# # Filter the blurred grayscale images using filter2D
# filtered_blurred_x = cv.filter2D(gray_blur, cv.CV_32F, sobel_x)
# filtered_blurred_y = cv.filter2D(gray_blur, cv.CV_32F, sobel_y)
#
# mag = cv.magnitude(filtered_blurred_x, filtered_blurred_y)
# orien = cv.phase(filtered_blurred_x, filtered_blurred_y, angleInDegrees=True)
# orien = orien / 2.  # Go from 0:360 to 0:180
# hsv = np.zeros_like(cv.resize(imageRGB, (imageIR.shape[1], imageIR.shape[0])))
# hsv[..., 0] = orien  # H (in OpenCV between 0:180)
# hsv[..., 1] = 255  # S
# hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  # V 0:255
#
# bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
# cv.imshow("Color coded edges", bgr)
# cv.waitKey(0)

# rgb = cv.resize(imageRGB.GRAYSCALE(), (imageIR.shape[1], imageIR.shape[0]))
# text = cv.resize(cv.pyrUp(cv.pyrUp(cv.pyrUp(cv.pyrDown(cv.pyrDown(cv.pyrDown(imageIR)))))), (imageIR.shape[1], imageIR.shape[0]))/255
# text_rgb = cv.resize(cv.pyrUp(cv.pyrUp(cv.pyrUp(cv.pyrDown(cv.pyrDown(cv.pyrDown(rgb)))))), (imageIR.shape[1], imageIR.shape[0]))/255
# detail = abs(imageIR/255 - text)
# detail = detail/detail.max()
# detail_rgb = (abs(rgb/255 - text))
# detail_rgb = detail_rgb/detail_rgb.max()
# cv.imshow('original', imageIR)
# cv.imshow('detail', detail)
# cv.imshow('texture', text)
# cv.imshow('original RGB', rgb)
# cv.imshow('detail RGB', detail_rgb)
# cv.imshow('texture RGB', text_rgb)

# cv.imshow('seg', segment)
# cv.waitKey(0)
# cv.destroyAllWindows()
# #
# ir = np.transpose(np.array((imageIR, imageIR, imageIR)), (1, 2, 0))
# print(ir.shape)
# #
# imageIR_text, imageIR_grad = Harr_pyr(imageIR, level=0, verbose=False, rebuilt=False)
# imageIR_grad, imageIR_text = imageIR_grad[0], imageIR_text[0]
# imageIR_grad = imageIR_grad/imageIR_grad.max()*255
#
# imageRGB_text, imageRGB_grad = Harr_pyr(imageRGB.GRAYSCALE(), level=0, verbose=False, rebuilt=False)
# imageRGB_grad, imageRGB_text = imageRGB_grad[0], imageRGB_text[0]
# imageRGB_grad = imageRGB_grad/imageRGB_grad.max()*255
#
# imageIR_fake_rgb = np.zeros([imageIR.shape[0], imageIR.shape[1], 3])
# imageIR_fake_rgb[:, :, 0] = imageIR
# imageIR_fake_rgb[:, :, 1] = imageIR_text*imageIR
# imageIR_fake_rgb[:, :, 2] = imageIR_grad
#
# imageRGB_fake_rgb = np.zeros([imageRGB.shape[0], imageRGB.shape[1], 3])
# imageRGB_fake_rgb[:, :, 0] = imageRGB.GRAYSCALE()
# imageRGB_fake_rgb[:, :, 1] = imageRGB_text*imageRGB.GRAYSCALE()
# imageRGB_fake_rgb[:, :, 2] = imageRGB_grad
#
# imageIR_fake_rgb = ImageCustom(imageIR_fake_rgb)
# imageIR_fake_rgb.cmap = 'HSV'
# imageRGB_fake_rgb = ImageCustom(imageRGB_fake_rgb)
# imageRGB_fake_rgb.cmap = 'HSV'
#
# cv.imshow('Fake image IR', imageIR_fake_rgb.RGB())
# cv.imshow('Fake image RGB', imageRGB_fake_rgb.RGB())
# cv.waitKey(0)
# cv.destroyAllWindows()

#
# temp = np.zeros([imageIR.shape[0], imageIR.shape[1], 3])
# temp[:, :, 0] = imageIR
# temp[:, :, 1] = imageIR
# temp[:, :, 2] = imageIR
# temp = ImageCustom(temp, imageIR)
# start = time.time()
# imgray = ImageCustom(cv.pyrDown(imageRGB.GRAYSCALE()))
# rgb = cv.pyrDown(imageRGB)
