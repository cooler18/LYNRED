import sys
import os
import sys

import numpy as np
from tqdm import tqdm

from FUSION.classes.Metrics import Metric_nrmse, Metric_ssim, Metric_nmi, Metric_psnr, Metric_rmse, Metric_nec
from FUSION.tools.gradient_tools import grad

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
from FUSION.script.image_management import name_generator
from FUSION.tools.registration_tools import *
from FUSION.classes.Image import ImageCustom

# t = 'Day'
# n = "00060"
# imgL = ImageCustom("/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + t + "/hybrid/infrared_projected/left" + n + ".png")
# imgR = ImageCustom("/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + t + "/hybrid/right/right" + n + ".png")
# imgL_original = ImageCustom("/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + t + "/hybrid/left/left" + n + ".png")
#         #######################
# # imgL = ImageCustom("/home/godeta/PycharmProjects/LYNRED/Images/Day/master/infrared_corrected/1594113918028_03_FR381940-03-0002.png")
# # imgR = ImageCustom("/home/godeta/PycharmProjects/LYNRED/Images/Day/slave/visible/1594113916853_03_4103616529.png")
# a = grad(imgR)
# b = grad(imgL)
# c = grad(imgL_original)
Time = ['Day']
percent = False
num = np.uint8(np.linspace(0, 99, 100))
rmse = {'ref': [],
        'new': []}
ssim = {'ref': [],
        'new': []}
nmi = {'ref': [],
       'new': []}
psnr = {'ref': [],
        'new': []}
nec = {'ref': [],
       'new': []}

for t in Time:
    for n in tqdm(num):
        number = name_generator(n)
        # print(f"number : " + number)
        imgL = ImageCustom(
            "/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + t + "/hybrid/infrared_projected/left" + number + ".png")
        imgR = ImageCustom(
            "/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + t + "/hybrid/right/right" + number + ".png")
        imgL_original = ImageCustom(
            "/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + t + "/hybrid/left/left" + number + ".png")
        #######################
        a = grad(imgR)
        b = grad(imgL)
        c = grad(imgL_original)
        rmse['ref'].append(Metric_rmse(a, c).value)
        rmse['new'].append(Metric_rmse(a, b).value)
        ssim['ref'].append(Metric_ssim(a, c).value)
        ssim['new'].append(Metric_ssim(a, b).value)
        nmi['ref'].append(Metric_nmi(a, c).value)
        nmi['new'].append(Metric_nmi(a, b).value)
        psnr['ref'].append(Metric_psnr(a, c).value)
        psnr['new'].append(Metric_psnr(a, b).value)
        nec['ref'].append(Metric_nec(a, c).value)
        nec['new'].append(Metric_nec(a, b).value)

#### RMSE ######
rmse['ref'] = np.array(rmse['ref'])
rmse['new'] = np.array(rmse['new'])
if percent:
    temp = (rmse['new'] - rmse['ref']) / (rmse['ref']) * 100
else:
    temp = (rmse['new'] - rmse['ref'])
mean_rmse = np.mean(temp)
var_rmse = np.std(temp)
max_rmse = np.min(temp)
min_rmse = np.max(temp)
argmax_rmse = np.argmin(temp)
argmin_rmse = np.argmax(temp)
###############

#### NMI ######
nmi['ref'] = np.array(nmi['ref'])
nmi['new'] = np.array(nmi['new'])
if percent:
    temp = (nmi['new'] - nmi['ref']) / (nmi['new']) * 100
else:
    temp = (nmi['new'] - nmi['ref'])
mean_nmi = np.mean(temp)
var_nmi = np.std(temp)
max_nmi = np.max(temp)
min_nmi = np.min(temp)
argmax_nmi = np.argmax(temp)
argmin_nmi = np.argmin(temp)
###############

#### PSNR ######
psnr['ref'] = np.array(psnr['ref'])
psnr['new'] = np.array(psnr['new'])
if percent:
    temp = (psnr['new'] - psnr['ref']) / (psnr['ref']) * 100
else:
    temp = (psnr['new'] - psnr['ref'])
mean_psnr = np.mean(temp)
var_psnr = np.std(temp)
max_psnr = np.max(temp)
min_psnr = np.min(temp)
argmax_psnr = np.argmax(temp)
argmin_psnr = np.argmin(temp)
###############

#### SSIM ######
ssim['ref'] = np.array(ssim['ref'])
ssim['new'] = np.array(ssim['new'])
if percent:
    temp = (ssim['new'] - ssim['ref']) / (ssim['ref']) * 100
else:
    temp = (ssim['new'] - ssim['ref'])
mean_ssim = np.mean(temp)
var_ssim = np.std(temp)
max_ssim = np.max(temp)
min_ssim = np.min(temp)
argmax_ssim = np.argmax(temp)
argmin_ssim = np.argmin(temp)
###############

#### NEC ######
nec['ref'] = np.array(nec['ref'])
nec['new'] = np.array(nec['new'])
if percent:
    temp = (nec['new'] - nec['ref']) / (nec['ref']) * 100
else:
    temp = (nec['new'] - nec['ref'])
mean_nec = np.mean(temp)
var_nec = np.std(temp)
max_nec = np.max(temp)
min_nec = np.min(temp)
argmax_nec = np.argmax(temp)
argmin_nec = np.argmin(temp)
###############


print(f'rmse mean : {mean_rmse}, max : {max_rmse} at {argmax_rmse}, min : {min_rmse} at {argmin_rmse}, variance : {var_rmse}')
print(f'nmi mean : {mean_nmi}, max : {max_nmi} at {argmax_nmi}, min : {min_nmi} at {argmin_nmi}, variance : {var_nmi}')
print(f'psnr mean : {mean_psnr}, max : {max_psnr} at {argmax_psnr}, min : {min_psnr} at {argmin_psnr}, variance : {var_psnr}')
print(f'ssim mean : {mean_ssim}, max : {max_ssim} at {argmax_ssim}, min : {min_ssim} at {argmin_ssim}, variance : {var_ssim}')
print(f'nec mean : {mean_nec}, max : {max_nec} at {argmax_nec}, min : {min_nec} at {argmin_nec}, variance : {var_nec}')

# #
# print("original similarity IR/RGB", Metric_nrmse(c, a))
# print("corrected similarity IR/RGB", Metric_nrmse(b, a))
# print("original similarity IR/RGB", Metric_rmse(c, a))
# print("corrected similarity IR/RGB", Metric_rmse(b, a))
# print("original similarity IR/RGB", Metric_ssim(c, a))
# print("corrected similarity IR/RGB", Metric_ssim(b, a))
# print("original similarity IR/RGB", Metric_nmi(c, a))
# print("corrected similarity IR/RGB", Metric_nmi(b, a))
# print("original similarity IR/RGB", Metric_psnr(c, a))
# print("corrected similarity IR/RGB", Metric_psnr(b, a))
# print("original similarity IR/RGB", Metric_nec(c, a))
# print("corrected similarity IR/RGB", Metric_nec(b, a))
#
# cv.imshow('Visible', a)
# cv.imshow('Infrared', b)
# cv.imshow('Original Infrared', c)
# cv.waitKey(0)
# cv.destroyAllWindows()

# print("original similarity IR/RGB", Metric_nrmse(imgL_original, imgR.LAB()[:, :, 0]))
# print("corrected similarity IR/RGB", Metric_nrmse(imgL, imgR.LAB()[:, :, 0]))
# print("original similarity IR/RGB", Metric_ssim(imgL_original, imgR.LAB()[:, :, 0]))
# print("corrected similarity IR/RGB", Metric_ssim(imgL, imgR.LAB()[:, :, 0]))
# print("original similarity IR/RGB", Metric_nmi(imgL_original, imgR.LAB()[:, :, 0]))
# print("corrected similarity IR/RGB", Metric_nmi(imgL, imgR.LAB()[:, :, 0]))
# print("original similarity IR/RGB", Metric_psnr(imgL_original, imgR.LAB()[:, :, 0]))
# print("corrected similarity IR/RGB", Metric_psnr(imgL, imgR.LAB()[:, :, 0]))
# print("original similarity IR/RGB", Metric_edge_correlation(imgL_original, imgR.LAB()[:, :, 0]))
# print("corrected similarity IR/RGB", Metric_edge_correlation(imgL, imgR.LAB()[:, :, 0]))
