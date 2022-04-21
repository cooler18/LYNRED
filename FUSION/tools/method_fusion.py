import matplotlib.pyplot as plt

from FUSION.tools.data_management_tools import register_cmap_Lynred
from FUSION.tools.gradient_tools import edges_extraction, Harr_pyr
from FUSION.tools.manipulation_tools import *
import numpy as np
from matplotlib import cm
from FUSION.classes.Image import ImageCustom
from skimage import filters, morphology
from FUSION.tools import mapping_tools as m_tools


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
    if domain == 'LAB':
        rgb = rgb.LAB()
    elif domain == 'HSV':
        rgb = rgb.HSV()
    fusion = np.zeros_like(rgb)
    if domain == 'HSV':
        fusion[:, :, 0] = rgb[:, :, 0]
        fusion[:, :, 1] = rgb[:, :, 1]
        fusion[:, :, 2] = fusion_scaled(gray, rgb[:, :, 2], ratio=ratio)
        fusion = ImageCustom(fusion, rgb).RGB()
    if domain == 'LAB':
        fusion[:, :, 1] = rgb[:, :, 1]
        fusion[:, :, 2] = rgb[:, :, 2]
        fusion[:, :, 0] = fusion_scaled(gray, rgb[:, :, 0], ratio=ratio)
        fusion = fusion.RGB()
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
