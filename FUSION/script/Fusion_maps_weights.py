import os
import sys

import numpy as np

from FUSION.classes.Mask import Mask
from FUSION.classes.Metrics import *
from FUSION.script.image_management import name_generator

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname((os.path.dirname(SCRIPT_DIR))))
from FUSION.tools.method_fusion import *
from FUSION.tools.registration_tools import *

number = -1
Time = 'Day'
if number == -1:
    number = name_generator(np.random.randint(0, 99))

im_type = 'visible'
print(f"number : " + number)
imgL = ImageCustom("/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + Time + "/hybrid/infrared_projected/left" + number + ".png")
imgR = ImageCustom("/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + Time + "/hybrid/right/right" + number + ".png")
imgL_original = ImageCustom("/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + Time + "/hybrid/left/left" + number + ".png")
fus = imgR.LAB()
fus2 = imgR.LAB()
ref = imgR.LAB()
#######################
weightRGB = 1
#######################
mask = Mask(imgR, imgL)
mask_ssim = mask.ssim(weightRGB=weightRGB)
mask_saliency = mask.saliency(verbose=False, weightRGB=weightRGB, intensityRBG=1, intensityIR=1, colorRGB=1, edgeRGB=1, edgeIR=1)
fus[:, :, 0] = fus[:, :, 0] * mask_ssim + imgL * (1-mask_ssim)
fus2[:, :, 0] = fus2[:, :, 0] * mask_saliency + imgL * (1-mask_saliency)
# fus2[:, :, 1][imgR.LAB()[:, :, 0]<10] = 128 + (fus2[:, :, 1][imgR.LAB()[:, :, 0]<10]/1.0 - 128) * 1.2
# fus2[:, :, 2][imgR.LAB()[:, :, 0]<10] = 128 + (fus2[:, :, 2][imgR.LAB()[:, :, 0]<10]/1.0 - 128) * 1.2

ref[:, :, 0] = ref[:, :, 0] * (weightRGB/(weightRGB + 1)) + imgL*(1- weightRGB/(weightRGB + 1))

results = np.hstack([fus.BGR(), fus2.BGR(), ref.BGR()])
images = np.hstack([mask.RGB.BGR(), mask.IR.RGB('gray')])
results_gray = np.hstack([fus.GRAYSCALE(), fus2.GRAYSCALE(), ref.GRAYSCALE()])
masks = np.hstack([mask_ssim, mask_saliency])
color_layer = np.hstack([imgR.LAB()[:, :, 1], imgR.LAB()[:, :, 2]])


print("New value of similarity RGB/FUS", Metric_nrmse(imgR.LAB()[:, :, 0], fus[:, :, 0]))
print("New value of similarity IR/FUS", Metric_nrmse(imgL, fus[:, :, 0]))
print("Total value of similarity IR/RGB", Metric_nrmse(imgL, fus[:, :, 0]) + Metric_nrmse(imgR.LAB()[:, :, 0], fus[:, :, 0]))

print("\nNew value of saliency RGB/FUS", Metric_nrmse(imgR.LAB()[:, :, 0], fus2[:, :, 0]))
print("New value of saliency IR/FUS", Metric_nrmse(imgL, fus2[:, :, 0]))
print("Total value of similarity IR/RGB", Metric_nrmse(imgL, fus2[:, :, 0]) + Metric_nrmse(imgR.LAB()[:, :, 0], fus2[:, :, 0]))

print("\nNew value of ref RGB/FUS", Metric_nrmse(imgR.LAB()[:, :, 0], ref[:, :, 0]))
print("New value of ref IR/FUS", Metric_nrmse(imgL, ref[:, :, 0]))
print("Total value of 1/2", Metric_nrmse(imgL, ref[:, :, 0])+Metric_nrmse(imgR.LAB()[:, :, 0], ref[:, :, 0]))


cv.imshow('Result : SSIM mask, Saliency Mask, 1/2 Mask', results)
cv.imshow('Result gray : SSIM mask, Saliency Mask, 1/2 Mask', results_gray)
cv.imshow('images', images)
cv.imshow('SSIM mask, Saliency Mask', masks)
cv.imshow('color layers', color_layer)
cv.waitKey(0)
cv.destroyAllWindows()