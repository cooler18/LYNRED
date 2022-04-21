# from FUSION.classes.Image import *
import math
from math import atan2, cos, sqrt, sin, pi

import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import skimage
# from scipy import ndimage
from scipy.ndimage import median_filter, uniform_filter, generic_filter
# from scipy.stats import stats
from skimage.filters.rank import majority
# from sklearn.decomposition import PCA
from scipy.interpolate import interp2d, CloughTocher2DInterpolator
import scipy.interpolate as interpolate
from FUSION.interface.interface_tools import random_image_opening, prepare_image
from FUSION.tools.gradient_tools import *
# from skimage import filters, morphology
from FUSION.tools.manipulation_tools import *
# from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.interpolate import griddata


def densification_by_interpolation(maps, intensity=2, method='Default', verbose=True):
    start = time.time()
    r = np.linspace(0, maps.shape[0]-1, round(maps.shape[0]/5), dtype=int)
    c = np.linspace(0, maps.shape[1]-1, round(maps.shape[1]/5), dtype=int)
    rr, cc = np.meshgrid(r, c)
    vals = np.nonzero((maps[rr, cc] < 1) * (maps[rr, cc] > 0))
    # vals = np.nonzero((maps < 1) * (maps > 0))
    # with np.printoptions(threshold=np.inf):
    #     print(maps[rr, cc][vals].shape)
    ex = np.zeros_like(maps)
    '''
    If the method choose is the interpolation
    '''
    if method == 'interpolation':
        dst = maps.copy()
        f = interpolate.RBFInterpolator(np.transpose(vals), maps[rr, cc][vals], neighbors=100, smoothing=1.0,
                                        kernel='linear', epsilon=None, degree=0)
        mask = np.nonzero((maps > 1-intensity/100) + (maps < intensity/100))
        res = f(np.transpose(mask))
        dst[mask] = res
        ex[mask] = res

    '''
    If the method choose is the CloughTorcher2DInterpolator
    '''
    if method == 'CloughTorcher':
        interp = CloughTocher2DInterpolator(vals, maps[vals])
        dst = maps.copy()
        res = interp(np.nonzero((maps > 1-intensity/100) + (maps < intensity/100)))
        dst[(maps > 1-intensity/100) + (maps < intensity/100)] = res
        ex[(maps > 1-intensity/100) + (maps < intensity/100)] = res
    '''
    If the method choose is the griddata interpolation
    '''
    if method == 'griddata':
        dst = maps.copy()
        res = griddata(np.transpose(vals), maps[vals], np.nonzero((maps > 1-intensity/100) + (maps < intensity/100)), method='linear', rescale=False)
        dst[(maps > 1-intensity/100) + (maps < intensity/100)] = res
        ex[(maps > 1-intensity/100) + (maps < intensity/100)] = res

    '''
    If the method choose is the Inpainting by OpenCV
    '''
    if method == "inpainting" or method == "Default":
        mask = np.uint8(np.array(((maps > 1-intensity/100) + (maps < intensity/100)) * 255))
        dst = cv.inpaint(maps, mask, 20, cv.INPAINT_NS)
        ex[(maps > 1-intensity/100) + (maps < intensity/100)] = dst[(maps > 1-intensity/100) + (maps < intensity/100)]
    print(f"Done in : {time.time() - start} seconds")
    if verbose:
        cv.imshow('Old disparity map', maps)
        cv.imshow('New disparity map', dst)
        cv.imshow('Changes', ex)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return dst


def map_contrast(image, method="prewitt", kernel_size=5, kernel_blur=3, low_threshold=10, ratio=3):
    # calculation of the norm of the gradient of the image using prewitt method.
    # norm of the gradient is a measure of contrast

    im = edges_extraction(image.copy(), method=method, kernel_size=kernel_size, kernel_blur=kernel_blur,
                          low_threshold=low_threshold, ratio=ratio)
    im = ImageCustom(im, image)
    im.cmap = 'EDGES'
    return im


def template_matching(image_RGB, image_IR, split=(2, 2), method=cv2.TM_CCOEFF, L=50):
    # start = time.time()
    stepx = int(image_IR.shape[0] // split[0])
    stepy = int(image_IR.shape[1] // split[1])
    im = np.zeros_like(image_IR)
    mat = np.zeros([split[0], split[1]])
    # Load the target image with the shapes we're trying to match
    for i in range(split[0]):
        for j in range(split[1]):
            template = image_IR[stepx * i:stepx * (i + 1), stepy * j:stepy * (j + 1)]
            target = image_RGB[stepx * i:stepx * (i + 1), max(stepy * j - L, 0):min(stepy * (j + 1) + L, image_IR.shape[1])]
            s = target.shape[1] - L - stepy
            match = cv2.matchTemplate(target, template, method)
            delta = cv2.minMaxLoc(match)[3][0] - s
            if j != split[1] - 1:
                im[stepx * i:stepx * (i + 1), stepy * j + delta:stepy * (j + 1) + delta] = template
            elif delta > 0:
                im[stepx * i:stepx * (i + 1), stepy * j + delta:-(image_IR.shape[1] - stepy * (j + 1))] = \
                    template[:, :-delta]
            else:
                im[stepx * i:stepx * (i + 1), stepy * j + delta: stepy * (j + 1) + delta] = template
            mat[i, j] = delta
    return ImageCustom(im), mat


def orientation_calibration(method='prewitt', L=100, image_number=10, one=False, image_IR=0, image_RGB=0, tot=None,
                            verbose=False):
    match = {}
    ############ Loop to find the matched template ##################
    if one:
        if tot is None:
            if L == 0:
                tot = np.ones([image_RGB.shape[0] - image_IR.shape[0] + 1, image_RGB.shape[1] - image_IR.shape[1] + 1])
            else:
                tot = np.ones([L + 1, L + 1])
        ir = edges_extraction(image_IR, method=method, kernel_size=4, kernel_blur=1)
        if L == 0:
            template = ir.copy()
            target = edges_extraction(ImageCustom(image_RGB.GRAYSCALE()), method=method, kernel_size=4,
                                      kernel_blur=1)
        else:
            l = int(L / 2)
            template = ir[l:-l, l:-l]
            target = edges_extraction(ImageCustom(cv.pyrDown(image_RGB.GRAYSCALE())), method=method, kernel_size=4,
                                      kernel_blur=1)
        match = cv.matchTemplate(target, template, cv.TM_CCOEFF)
        match = match / match.max()
        tot = tot * match
    else:
        for i in range(image_number):
            print(f'image {i + 1} over {image_number}')
            vis, ir, _ = random_image_opening(verbose=0)
            image_RGB = ImageCustom(vis)
            image_IR = ImageCustom(ir)
            image_IR, image_RGB = size_matcher(image_IR, image_RGB)
            ir = edges_extraction(image_IR, method=method, kernel_size=4, kernel_blur=1)
            if L == 0:
                template = ir.copy()
                target = edges_extraction(ImageCustom(image_RGB.GRAYSCALE()), method=method, kernel_size=4,
                                          kernel_blur=1)
            else:
                tot = np.ones([L + 1, L + 1])
                l = int(L / 2)
                template = ir[l:-l, l:-l]
                target = edges_extraction(ImageCustom(cv.pyrDown(image_RGB.GRAYSCALE())), method=method, kernel_size=4,
                                          kernel_blur=1)
            match[i] = cv.matchTemplate(target, template, cv.TM_CCOEFF)
            match[i] = match[i] / match[i].max()
            tot = tot * match[i]
    tot = tot / tot.max()
    ########################## Compute the direction of the average of Number_images of images ###################
    angle = getOrientation(tot, acc_degree=0.1, verbose=verbose)
    tot1 = cv.pyrUp(cv.pyrUp(tot))
    center = np.where(tot1 == tot1.max())
    center = (center[1][0], center[0][0])
    p1 = (0, center[1] - center[0] * math.tan(math.radians(angle)))
    p2 = (tot1.shape[1] - 1, center[1] + (tot1.shape[1] - 1 - center[0]) * math.tan(math.radians(angle)))
    color = (255, 255, 255)
    u = np.uint8(np.linspace(p1[1], p2[1], tot1.shape[0]))
    v = np.uint8(np.linspace(p1[0], p2[0], tot1.shape[0]))
    axe = tot1[u, v]
    axe[axe < 0.75] = 0
    delta = np.linspace(-l, l, tot1.shape[0])
    start = round(delta[np.where(axe > 0)[0][0]]) - 1
    stop = round(delta[np.where(axe > 0)[0][-1]]) + 1
    if verbose:
        plt.plot(delta, axe)
        plt.show()
        print(f'The orientation of the entraxe is {angle} degrees from horizontal\n'
              f'Between {start} and {stop} pixels')
        cv.line(tot1, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1, cv.LINE_AA)
        cv2.circle(tot1, center, 3, color, 1)
        cv.imshow('Orientation of the Shift', tot1)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return angle, (start, stop), (center[0] / 4, center[1] / 4), tot


def orientation_calibration_cam(image_IR, image_RGB, method='prewitt', tot=None):
    match = {}
    ############ Loop to find the matched template ##################
    ir = ImageCustom(cv.pyrDown(image_IR))
    rgb = ImageCustom(cv.pyrDown(image_RGB.GRAYSCALE()))
    template = edges_extraction(ir, method=method, kernel_size=4, kernel_blur=1)
    target = edges_extraction(rgb, method=method, kernel_size=4, kernel_blur=1)
    match = cv.matchTemplate(target, template, cv.TM_CCOEFF)
    match = match / match.max()
    match = cv.pyrUp(match)
    tot = tot * match
    tot = tot / tot.max()
    ########################## Compute the direction of the average of Number_images of images ###################
    angle = getOrientation(tot, acc_degree=0.5, verbose=False)
    center = np.where(tot == tot.max())
    center = (center[0][0], center[1][0])
    a = abs(np.tan(math.radians(angle)))
    b = abs(np.tan(math.radians(90 - angle)))
    if (tot.shape[0] - center[0]) >= center[1] * a:
        p1 = (center[0] + center[1] * a - 1, 0)
    else:
        p1 = (tot.shape[0] - 1, center[1] - (tot.shape[0] - center[0]) * b)
    if a * (tot.shape[1] - center[1]) >= center[0]:
        p2 = (0, center[1] + center[0] * b - 1)
    else:
        p2 = (center[0] - (tot.shape[1] - center[1]) * a, tot.shape[1] - 1)
    u = np.uint8(np.linspace(p1[0], p2[0], tot.shape[0]))
    v = np.uint8(np.linspace(p1[1], p2[1], tot.shape[0]))
    axe = tot[u, v]
    axe[axe < 0.05] = 0
    dist_p1 = np.sqrt((p1[0] - center[0]) ** 2 + (p1[1] - center[1]) ** 2)
    dist_p2 = np.sqrt((p2[0] - center[0]) ** 2 + (p2[1] - center[1]) ** 2)
    delta = np.linspace(-dist_p1, dist_p2, tot.shape[0])
    start = round(delta[np.where(axe > 0)[0][0]]) - 1
    stop = round(delta[np.where(axe > 0)[0][-1]]) + 1
    return angle, (start, stop), (center[0], center[1]), tot


def map_distance(start, stop, angle, image_ir, image_rgb, level=4,
                 method='Prewitt', median=None, threshold=128, level_pyr=2, l_th=15, ratio=3, blur_filter=3,
                 disparity_left=True, return_images=False, gradient_scaled=True, uniform=True):
    angle = np.radians(angle)
    ## For level=11, every single pixel step is computed
    if level == 11:
        level = stop - start + 1
    rgb_small = cv.pyrDown(image_rgb.GRAYSCALE())
    if l_th:
        l_th_ir = int(l_th * image_ir.max() / 255)
        l_th_vis = int(l_th * rgb_small.max() / 255)
    else:
        l_th_vis, l_th_ir = None, None
    if method == 'Canny' or method == 'Perso':
        _, orient_ir = edges_extraction(image_ir,
                                        method='Prewitt', kernel_blur=blur_filter, low_threshold=l_th_ir,
                                        ratio=ratio,
                                        level=level_pyr, orientation=True)
        _, orient_rgb = edges_extraction(ImageCustom(rgb_small),
                                         method='Prewitt', kernel_blur=blur_filter, low_threshold=l_th_vis,
                                         ratio=ratio,
                                         level=level_pyr, orientation=True)
        edge_ir, _ = edges_extraction(image_ir,
                                      method=method, kernel_blur=blur_filter, low_threshold=l_th_ir, ratio=ratio,
                                      level=level_pyr, orientation=True)
        edge_rgb, _ = edges_extraction(ImageCustom(rgb_small),
                                       method=method, kernel_blur=blur_filter, low_threshold=l_th_vis,
                                       ratio=ratio,
                                       level=level_pyr, orientation=True)
    else:
        edge_ir, orient_ir = edges_extraction(image_ir,
                                              method=method, kernel_blur=blur_filter, low_threshold=l_th_ir,
                                              ratio=ratio,
                                              level=level_pyr, orientation=True)
        edge_rgb, orient_rgb = edges_extraction(ImageCustom(rgb_small),
                                                method=method, kernel_blur=blur_filter, low_threshold=l_th_vis,
                                                ratio=ratio,
                                                level=level_pyr, orientation=True)
    if gradient_scaled:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        edge_ir = clahe.apply(edge_ir)
        edge_rgb = clahe.apply(edge_rgb)
        if edge_ir.max():
            edge_ir = edge_ir / edge_ir.max() * 255
        if edge_rgb.max():
            edge_rgb = edge_rgb / edge_rgb.max() * 255
    edge_ir[edge_ir < threshold] = 0
    edge_rgb[edge_rgb < threshold] = 0
    edge_ir = edge_ir * 1.
    edge_rgb = edge_rgb * 1.
    # rgb_small, image_ir = (rgb_small > threshold)*255, (image_ir > threshold)*255
    idx = np.linspace(start, stop, level)
    y_step = np.array([round(dx * math.sin(math.radians(angle))) for dx in idx])
    x_step = np.array([round(dx * math.cos(math.radians(angle))) for dx in idx])
    temp = edge_correlation(edge_ir, edge_rgb, idx, x_step, y_step, orient_ir, orient_rgb, level, method,
                            disparity_left)
    mask = np.zeros([max(y_step.max() - y_step.min(), 1), max(x_step.max() - x_step.min(), 1)])
    temp = cv2.normalize(np.sqrt(temp), None, 0, 255, cv2.NORM_MINMAX)
    # base_maps = generate_support_map(temp.shape[:2], 0.4, min_slide=10, max_slide=255)
    temp[temp < threshold] = -1
    if uniform:
        temp2 = uniform_filter(temp, (round(mask.shape[0] / 2 + 1), round(mask.shape[1] / 2 + 1), 1), mode='reflect',
                               cval=0.0, origin=0)
    else:
        temp2 = cv.GaussianBlur(temp, (mask.shape[0] * 2 + 1, mask.shape[1] * 2 + 1), 1)
    temp = temp2 * (temp > 0)
    # maps = findMax(temp)#, mask.shape[0], mask.shape[1])
    maps = idx[np.argmax(temp, axis=2)]
    maps = np.uint8(cv2.normalize(maps, None, 0, 255, cv2.NORM_MINMAX))
    if median > 1:
        neighbor = np.ones([median, median])
        maps = majority(maps, footprint=neighbor)  # median_filter(maps, size=median)
    if disparity_left:
        maps[edge_ir == 0] = 0
    else:
        maps[edge_rgb == 0] = 0
    # maps[maps == 0] = base_maps[maps == 0]
    if return_images:
        return ImageCustom(maps), ImageCustom(edge_ir), ImageCustom(edge_rgb)
    return ImageCustom(maps)


def findMax(im, filter_x=3, filter_y=3, stride_x=10, stride_y=10):
    maps = np.zeros([im.shape[0], im.shape[1]])
    for i in range(0, maps.shape[0], stride_x):
        for j in range(0, maps.shape[1], stride_y):
            patch = im[i:i + filter_x, j:j + filter_y, :]
            m = patch.sum(axis=(0, 1)) / ((patch > 0).sum(axis=(0, 1)) + 1)
            maps[i:i + stride_x, j:j + stride_y] = np.ones_like(maps[i:i + stride_x, j:j + stride_y]) * np.argmax(m)
    return maps


def getOrientation(img, acc_degree=1., acc_rad=0, verbose=False):
    center = np.where(img == img.max())
    center = (float(center[0][0]), float(center[1][0]))
    rot = []
    if acc_rad:
        for i in np.linspace(-90, 90, int(acc_rad / pi * 180)):
            temp = rotate(img, i, center=center)
            rot.append(temp.sum(axis=0).max())
        if verbose:
            fig, ax = plt.figure()
            ax.plot(np.linspace(-90, 90, round(180 / acc_degree) + 1), rot)
            plt.show()
        return max(rot * pi / 180)
    else:
        for i in np.linspace(-90, 90, round(180 / acc_degree) + 1):
            temp = rotate(img, i, center=center)
            rot.append(temp.sum(axis=1).max())
        rot = np.array(rot)
        return np.linspace(-90, 90, round(180 / acc_degree) + 1)[np.argmax(rot)]


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def generate_support_map(size, horizon, min_slide=1, max_slide=255, **kwargs):
    maps = np.expand_dims(np.linspace(0, size[0] - 1, size[0]), axis=1)
    maps = np.dot(maps, np.ones([1, size[1]]))
    if horizon < 1:
        horizon = round(horizon * maps.shape[0])
    maps[horizon:, :] = np.tan(((maps[horizon:, :] - horizon) / (horizon - maps.shape[0]) + 1) * np.pi / 2 - 10 ** -1)
    maps[:horizon, :] = maps[horizon, 0]
    maps = cv.normalize(-maps, None, min_slide, max_slide, norm_type=cv.NORM_MINMAX)
    # if level > 1:
    #     maps = np.expand_dims(maps, axis=2)
    #     depth = np.ones([1, level])
    #     maps = np.dot(maps, depth)
    return np.round(maps)
