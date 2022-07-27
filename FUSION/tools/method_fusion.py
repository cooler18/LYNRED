import os
import sys

import matplotlib.pyplot as plt

from FUSION.tools.data_management_tools import register_cmap_Lynred
from FUSION.tools.gradient_tools import edges_extraction, Harr_pyr
from FUSION.tools.manipulation_tools import *
import numpy as np
from matplotlib import cm
from FUSION.classes.Image import ImageCustom
from . import __domain__
from .image_processing_tools import laplacian_fusion


def colormap_fusion(ir_gray, rgb, ratio=-1, colormap='inferno'):
    ir_rgb = ir_gray.RGB(colormap=colormap)
    if ratio != -1:
        return ImageCustom(fusion_scaled(ir_rgb, rgb, ratio=ratio)), ir_rgb
    else:
        rgb = rgb.GRAYSCALE()
        return ir_rgb, rgb.RGB(colormap=colormap)


def colormap_fusion2(ir_gray, rgb, ratio=0.5, colormap='inferno'):
    ir_rgb = ir_gray.RGB(colormap=colormap)
    return ImageCustom(fusion_scaled(ir_rgb, rgb, ratio=ratio))


def grayscale_fusion(gray, rgb, ratio=-1, domain='LAB'):
    rgb = __domain__[domain][0](rgb)
    idx = __domain__[domain][1]
    fusion = rgb.copy()
    fusion[:, :, idx] = fusion_scaled(gray, rgb[:, :, idx], ratio=ratio)
    fusion = ImageCustom(fusion, rgb).RGB()
    return fusion


def region_based_fusion(gray, rgb, method='Prewitt', ksize=5, kernel_blur=5, low_threshold=10, ratio=3, level=1):
    # gray_m, rgb_m = size_matcher(gray.copy(), rgb.copy())
    m_vis = edges_extraction(rgb, method=method, kernel_size=ksize, kernel_blur=kernel_blur,
                             low_threshold=low_threshold, ratio=ratio, level=level)
    m_ir = edges_extraction(gray, method=method, kernel_size=ksize, kernel_blur=kernel_blur,
                            low_threshold=low_threshold, ratio=ratio, level=level)
    return m_ir, m_vis


def Harr_fus(gray, rgb, ratio=0.5, level=1):
    gray, rgb = size_matcher(gray, rgb)
    rgb = rgb.LAB()
    gray_text, gray_detail = Harr_pyr(gray, level=level)
    rgb_text, rgb_detail = Harr_pyr(rgb[:, :, 0], level=level + 1)
    image = ratio * (gray_text[level] + gray_detail[level]) + (1 - ratio) * (
                rgb_text[level + 1] + rgb_detail[level + 1])
    # for i in range(level):
    #     image = cv.resize(cv.pyrUp(image), (gray_detail[level - i].shape[1], gray_detail[level - i].shape[0]))
    #     image = image + gray_detail[level - i] * ratio + rgb_detail[level - i + 1] * (1 - ratio)
    # print(image.min(), image.max())
    image = cv.resize(cv.pyrUp(image), (rgb_detail[0].shape[1], rgb_detail[0].shape[0])) + rgb_detail[0]
    image = image/image.max()*255
    image = np.stack([image, rgb[:, :, 1], rgb[:, :, 2]], axis=2)
    return ImageCustom(image, rgb).RGB()


def mask_fusion_intensity(RGB, IR):
    temp = np.ones_like(RGB)
    return ImageCustom(np.minimum(temp, RGB/255 - IR/255)*255).gaussian_filter(3)

def laplacian_pyr_fusion(image1, image2, mask, octave=4, verbose=False):
    if image1.shape[0] > image2.shape[0]:
        temp = image1.LAB()[:, :, 0]
        image_detail = ImageCustom(cv.pyrUp(cv.pyrDown(temp))).diff(temp)
        image1 = ImageCustom(cv.pyrDown(temp))
        image2 = ImageCustom(image2)
    elif image1.shape[0] < image2.shape[0]:
        temp = image2.LAB()[:, :, 0]
        image_detail = ImageCustom(cv.pyrUp(cv.pyrDown(temp)))/255 - temp/255
        image2 = ImageCustom(cv.pyrDown(temp))
        image1 = ImageCustom(image1)
    else:
        image_detail = None
    pyr1 = image1.GRAYSCALE().pyr_gauss(octave=octave, interval=2, sigma0=2)
    pyr2 = image2.GRAYSCALE().pyr_gauss(octave=octave, interval=2, sigma0=2)
    fus = laplacian_fusion(pyr1, pyr2, mask, verbose=verbose)
    if image_detail is not None:
        return ImageCustom(cv.pyrUp(fus)) + image_detail
    else:
        return fus

def mean(im1, im2):
    return ImageCustom(im1/2 + im2/2)

