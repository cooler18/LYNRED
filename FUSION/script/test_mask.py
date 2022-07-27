import os
import random
import sys
import time

from FUSION.classes.Mask import Mask
from FUSION.classes.Metrics import *
from FUSION.script.image_management import name_generator

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname((os.path.dirname(SCRIPT_DIR))))
from FUSION.tools.method_fusion import *
from FUSION.tools.registration_tools import *

Time = 'Day'
#"00090"  # #"00013"  #
number = "00090"  #name_generator(random.randint(0, 3600))
metric = Metric_ssim
weightRGB = 2
Laplacian_pyr = True

####################################
print(f"number : " + number)
imgL = ImageCustom(
    "/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + Time + "/hybrid/infrared_projected/left" + number + ".png")
imgR = ImageCustom("/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + Time + "/hybrid/right/right" + number + ".png")

# imgL = ImageCustom("/home/godeta/PycharmProjects/LYNRED/Images/ir.png")
# imgR = ImageCustom("/home/godeta/PycharmProjects/LYNRED/Images/vis.png")

# if not Laplacian_pyr:
#     imgR = ImageCustom(cv.pyrDown(imgR))
#     im_mask = imgR
# else:
im_mask = ImageCustom(cv.pyrDown(imgR))
fus = imgR.LAB()
fus2 = imgR.LAB()
fus3 = imgR.LAB()
fus4 = imgR.LAB()
ref = imgR.LAB()

##################################################
# b = np.transpose(np.expand_dims(np.linspace(0, imgR.shape[1], imgR.shape[1]), axis=1))
# a = np.ones([imgR.shape[0], 1])
# m = a * b
# m = 1-m/m.max()
m = np.ones_like(ref[:, :, 0])*weightRGB/(1+weightRGB)
mask = Mask(im_mask, imgL)
saliency = mask.static_saliency(weightRGB=weightRGB, verbose=True)
mask_ssim = mask.ssim(weightRGB=weightRGB, win_size=5)
saliency = (saliency*mask_ssim).mean_shift(weightRGB/(1+weightRGB))
mask_ssim_saliency = mask.saliency_ssim(verbose=False, level_max=1, weightRGB=weightRGB)
mask_saliency_gaussian = mask.saliency_gaussian(verbose=False, weightRGB=weightRGB, intensityRBG=1, intensityIR=1,
                                                colorRGB=1, edgeRGB=1, edgeIR=1)

mask_saliency_scale = mask.saliency_scale(verbose=False, weightRGB=weightRGB, intensityRBG=1, intensityIR=1, colorRGB=1,
                                          edgeRGB=1, edgeIR=1)

if Laplacian_pyr:
    fus[:, :, 0] = laplacian_pyr_fusion(fus, imgL, mask_ssim, octave=5)
    fus2[:, :, 0] = laplacian_pyr_fusion(fus2, imgL, mask_saliency_gaussian, octave=5)
    fus3[:, :, 0] = laplacian_pyr_fusion(fus3, imgL, mask_saliency_scale, octave=5)
    fus4[:, :, 0] = laplacian_pyr_fusion(fus4, imgL, saliency, octave=5)
    ref[:, :, 0] = laplacian_pyr_fusion(ref, imgL, m, octave=5)
else:
    fus[:, :, 0] = fus[:, :, 0]*mask_ssim + imgL*(1-mask_ssim)
    fus2[:, :, 0] = fus2[:, :, 0]*mask_saliency_gaussian + imgL*(1-mask_saliency_gaussian)
    fus3[:, :, 0] = fus3[:, :, 0]*mask_saliency_scale + imgL*(1-mask_saliency_scale)
    fus4[:, :, 0] = fus4[:, :, 0]*saliency + imgL*(1-saliency)
    ref[:, :, 0] = ImageCustom(ref[:, :, 0]*m) + imgL*np.ones_like(imgL)*(1-m)

print("\n############################ REFERENCE VALUE ###################################")
print("Ref value of similarity", metric(im_mask, imgL))

print("\n############################ SSIM ###################################")
print("New value RGB/FUS", metric(im_mask, fus[:, :, 0]))
print("New value IR/FUS", metric(imgL, fus[:, :, 0]))
print("Total value FUS/IR+RGB", metric(im_mask, imgL, fus))

print("\n############################ SALIENCY OPENCV ##############################")
print("New value RGB/FUS", metric(im_mask, fus4[:, :, 0]))
print("New value IR/FUS", metric(imgL, fus4[:, :, 0]))
print("Total value FUS/IR+RGB", metric(im_mask, imgL, fus4))

print("\n####################### SALIENCY GAUSSIAN ##############################")
print("New value RGB/FUS", metric(im_mask, fus2[:, :, 0]))
print("New value IR/FUS", metric(imgL, fus2[:, :, 0]))
print("Total value FUS/IR+RGB", metric(im_mask, imgL, fus2))

print("\n######################### SALIENCY SCALE ##############################")
print("New value RGB/FUS", metric(imgR, fus3[:, :, 0]))
print("New value IR/FUS", metric(imgL, fus3[:, :, 0]))
print("Total value FUS/IR+RGB", metric(imgR, imgL, fus3))

print("\n############################# REF 1/2 ##############################")
print("New value RGB/FUS", metric(im_mask, ref[:, :, 0]))
print("New value IR/FUS", metric(imgL, ref[:, :, 0]))
print("Total value FUS/IR+RGB", metric(im_mask, imgL, ref))

# p = '/home/godeta/Images/images_rapport/4th part/analyze'
# cv.imwrite(p + '/alpha/' + str(weightRGB) +'_full.png', ref.BGR())
# cv.imwrite(p + '/alpha/' + str(weightRGB) +'_detail1.png', ref.BGR()[:130, 150:420])
# cv.imwrite(p + '/alpha/' + str(weightRGB) +'_detail2.png', ref.BGR()[150:270, 200:330])
#
# cv.imwrite(p + '/ssim/' + str(weightRGB) +'_full.png', fus.BGR())
# cv.imwrite(p + '/ssim/' + str(weightRGB) +'_detail1.png', fus.BGR()[:130, 150:420])
# cv.imwrite(p + '/ssim/' + str(weightRGB) +'_detail2.png', fus.BGR()[150:270, 200:330])
#
# cv.imwrite(p + '/ssim_scale/' + str(weightRGB) +'_full.png', fus4.BGR())
# cv.imwrite(p + '/ssim_scale/' + str(weightRGB) +'_detail1.png', fus4.BGR()[:130, 150:420])
# cv.imwrite(p + '/ssim_scale/' + str(weightRGB) +'_detail2.png', fus4.BGR()[150:270, 200:330])
#
# cv.imwrite(p + '/sal_gauss/' + str(weightRGB) +'_full.png', fus2.BGR())
# cv.imwrite(p + '/sal_gauss/' + str(weightRGB) +'_detail1.png', fus2.BGR()[:130, 150:420])
# cv.imwrite(p + '/sal_gauss/' + str(weightRGB) +'_detail2.png', fus2.BGR()[150:270, 200:330])
#
# cv.imwrite(p + '/sal_scale/' + str(weightRGB) +'_full.png', fus3.BGR())
# cv.imwrite(p + '/sal_scale/' + str(weightRGB) +'_detail1.png', fus3.BGR()[:130, 150:420])
# cv.imwrite(p + '/sal_scale/' + str(weightRGB) +'_detail2.png', fus3.BGR()[150:270, 200:330])
#

# cv.imshow('RGB ssim image', ImageCustom(mask_ssim*255).RGB(colormap='coolwarm').BGR())
cv.imshow('fusion weighted SSIM', fus.BGR())
# cv.imshow('RGB Saliency image', ImageCustom(mask_saliency_gaussian*255).RGB(colormap='coolwarm').BGR())
# cv.imshow('RGB Saliency scale image', ImageCustom(mask_saliency_scale*255).RGB(colormap='coolwarm').BGR())
# cv.imshow('fusion weighted SALIENCY', fus2.BGR())
# cv.imshow('fusion weighted SALIENCY SCALE', fus3.BGR())
cv.imshow('fusion weighted SALIENCY OPENCV', fus4.BGR())
cv.imshow('RGB saliency image', ImageCustom(saliency*255).RGB(colormap='coolwarm').BGR())
# cv.imshow('fusion 1/2', ref.BGR())
# cv.imshow('RGB', ImageCustom(m*255).RGB(colormap='coolwarm').BGR())
# cv.imshow('IR', imgL)
cv.waitKey(0)
cv.destroyAllWindows()
