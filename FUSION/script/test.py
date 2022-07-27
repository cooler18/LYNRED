import os, sys
# from os.path import *
# import tkinter as tk

from FUSION.classes.Mask import Mask
from FUSION.classes.Metrics import *

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname((os.path.dirname(SCRIPT_DIR))))
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
from FUSION.tools.registration_tools import *

# import numpy as np
# from FUSION.interface.Application import Application
# import time
# from scipy.ndimage import median_filter
#
# from Stereo_matching.Tools.disparity_tools import reprojection_disparity


# with open("/home/godeta/PycharmProjects/LYNRED/Video_frame/Night/hybrid/Calibration/transform_matrix", "rb") as f:
#     disparity = np.array(pickle.load(f))
# print(disparity)
Time = 'Night'
number = "00090"#name_generator(random.randint(0, 99))
metric = Metric_rmse

print(f"number : " + number)
imgL = ImageCustom(
    "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/hybrid/left/left" + number + ".png")
imgR = ImageCustom("/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/hybrid/right/right" + number + ".png")

imgR = ImageCustom("/home/godeta/PycharmProjects/LYNRED/Images/Day/master/visible/1594113918043_03_4103616527.png")
imgL = ImageCustom("/home/godeta/PycharmProjects/LYNRED/Images/Day/master/visible/1594113918079_03_4103616527.png")
# imgR = ImageCustom("/home/godeta/Images/images_rapport/1st part/vis_ex.png")
# imgL = ImageCustom("/home/godeta/Images/images_rapport/2nd part/example_disparity_acv.png")
# fus = ImageCustom(cv.pyrDown(imgR.LAB()), imgR.LAB())
# fus[:, :, 1:] = imgL.LAB()[:, :, 1:]
# weightRGB = 1
pts_src, pts_dst = SIFT(imgL, imgR, MIN_MATCH_COUNT=4, matcher='FLANN', name='00', lowe_ratio=0,
         nfeatures=0, nOctaveLayers=4, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6, toggle_evaluate=False, verbose=False,
         delta=None)
warp_matrix_rotation, _ = cv.findHomography(pts_src, pts_dst)
print(warp_matrix_rotation)
m, n = imgL.shape[:2]
imgR_aligned = ImageCustom(cv.warpPerspective(imgL.copy(), warp_matrix_rotation, (n, m)))#, flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
mask = Mask(imgR_aligned, imgL)
diff = mask.DIFF
# pyr = imgR.pyr_gauss(octave=4, interval=1, sigma0=1, verbose=True)

cv.imshow('diff', diff.BGR())
cv.imshow('aligned', imgR_aligned.BGR())
cv.imshow('BGR', imgL.BGR())
# cv.imshow('IR', imgL)
cv.waitKey(0)
cv.destroyAllWindows()

# cv.imshow('BGR', fus.BGR())
# cv.imshow('tot', fus.BGR())

# mask = Mask(imgR, imgL)
# mask_ssim = mask.ssim(weightRGB=weightRGB)
# mask_saliency_gaussian = mask.saliency_gaussian(verbose=False, weightRGB=weightRGB, intensityRBG=1, intensityIR=1,
#                                                 colorRGB=1,
#                                                 edgeRGB=1, edgeIR=1)
# mask_saliency_scale = mask.saliency_scale(verbose=False, weightRGB=weightRGB, intensityRBG=1, intensityIR=1, colorRGB=1,
#                                           edgeRGB=1, edgeIR=1)
# fus[:, :, 0] = laplacian_pyr_fusion(fus[:, :, 0], imgL, mask_ssim, octave=4)
# fus2[:, :, 0] = laplacian_pyr_fusion(fus2[:, :, 0], imgL, mask_saliency_gaussian, octave=4)
# fus3[:, :, 0] = laplacian_pyr_fusion(fus3[:, :, 0], imgL, mask_saliency_scale, octave=4)
# ref[:, :, 0] = laplacian_pyr_fusion(ref[:, :, 0], imgL, np.ones_like(imgL)*(weightRGB/(1+weightRGB)), octave=4)
#

# cv.imshow('fus', fus[:, :, 0])
# cv.imshow('S', fus[:, :, 1])
# cv.imshow('V', fus[:, :, 2])
# cv.imshow('BGR', fus.BGR())
# cv.imshow('tot', fus.BGR())
# cv.imshow('fusion weighted SALIENCY', fus2.BGR())
# cv.imshow('fusion weighted SALIENCY SCALE Laplacian', fus3.BGR())
# cv.imshow('fusion 1/2', ref.BGR())
#
# cv.imshow('IR', mask.IR)
# cv.imshow('RGB', imgR.BGR())

# src = {"A": imgL,
#        "B": imgR.RGB()}
# test_i2v()
# R = imgR.copy()
# R[:, :, 0] = R[:, :, 0].mean_shift()
# L = imgL.mean_shift()

# edges = edges_extraction(L.copy(), method='Canny', kernel_size=3, kernel_blur=5,
#                           low_threshold=15, ratio=10)
# edges += edges_extraction(R.GRAYSCALE(), method='Canny', kernel_size=3, kernel_blur=7,
#                           low_threshold=11, ratio=18)
# mask = Mask(R.RGB(), L)
# mask_low = mask.low(low_threshold=50)
# mask_high = mask.high(high_threshold=255)
# mask_diff = (ImageCustom(mask.diff(threshold=5, gaussian=5))/255)
# im_with_keypoints = blobdetection(mask_high)

# im_temp = R.copy()
# print(im_temp.cmap)
# # im_temp[:, :, 0] = im_temp[:, :, 0] * (0.8 +0.2 * mask_diff)
# # im_temp[:, :, 0][mask_high > 0] = imgR[:, :, 0][mask_high > 0]*mask_high[mask_high > 0]
# im_temp[:, :, 0][mask_low > 0] = (L[mask_low > 0]*mask_low[mask_low > 0] +
#                                   R[:, :, 0][mask_low > 0]*(1-mask_low[mask_low > 0]))


########.
# cv.imwrite('test.png', im_temp.BGR())
# cv.imwrite('mask.png', np.uint8((mask_high > 0)*255))
# cv.imwrite('edges.png', edges*mask_diff)
# im_temp = ImageCustom(edgeconnect(mode=2))
#########
###########

# cv.imshow('mask low', np.uint8(mask_low*255))
# cv.imshow('image diff', np.uint8(mask.DIFF))
# cv.imshow('mask diff', np.uint8(mask_diff*255))
# cv.imshow('RGB original', imgR.BGR())
# cv.imshow('IR original', imgL)
# cv.imshow('RGB mask', np.uint8(mask.RGB.BGR()))
# cv.imshow('IR mask', np.uint8(mask.IR))
# cv.imshow('mask high', np.uint8(mask_high*255))
# cv.imshow('new_image', im_temp.BGR())
# cv.waitKey(0)
# cv.destroyAllWindows()
###########
# cv.imshow('image infrared', imgL)
# cv.imshow('image RGB', imgR.BGR())
# cv.imshow('mask low', np.uint8(mask_low*255))
# cv.imshow('mask diff', np.uint8(mask.DIFF))
# cv.imshow('RGB mask', np.uint8(mask.RGB.BGR()))
# cv.imshow('IR mask', np.uint8(mask.IR))
# cv.imshow('mask high', np.uint8(mask_high*255))
# cv.imshow('mask edges', edges)
cv.waitKey(0)
cv.destroyAllWindows()

#
# with open("/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/hybrid/new_disparity/disp" + number, "rb") as f:
#     new_disparity = np.array(pickle.load(f))/2
# depth = 100 / (new_disparity + 0.005)
# depth[depth > 50] = 50
# depth = 1 - depth/50
