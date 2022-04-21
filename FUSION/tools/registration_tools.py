import argparse
import time

import cv2 as cv
import timeit
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import area_closing, closing, square, cube, disk, ball
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, normalized_mutual_information


def reconstruction_from_disparity(imageL, imageR, disparity_matrix, min_disp, max_disp, scale, closing_bool, verbose,
                                  median, inpainting, orientation=0):
    """
            Reconstruction function from disparity map.
            :return: reconstruct left image.
        """
    start = time.time()
    # Disparity map hold between 0 and 1 with the min/max  disparity pass as arguments.
    if scale == 0:
        scale = 1
    elif scale <= 2:
        scale = 2 * scale
    else:
        scale = 1

    left2right = max_disp - min_disp < 0
    disp = np.round(disparity_matrix * (max_disp - min_disp) * scale + min_disp * scale)
    dx = int(abs(max_disp) * scale * np.cos(orientation))
    dy = int(abs(max_disp) * scale * np.sin(orientation))
    if left2right:
        if len(imageL.shape) > 2:
            image_proc = cv.cvtColor(imageL, cv.COLOR_BGR2GRAY)
        else:
            image_proc = imageL
    else:
        if len(imageR.shape) > 2:
            image_proc = cv.cvtColor(imageR, cv.COLOR_BGR2GRAY)
        else:
            image_proc = imageR
    new_image = np.zeros([imageL.shape[0] * scale + dy, imageL.shape[1] * scale + dx])

    if left2right:
        step = -1
    else:
        step = 1
    for k in range(round(min_disp * scale), round(max_disp * scale), step):
        delta = k - round(min_disp * scale)
        # print(f"k : {k}, delta : {delta}, dx : {dx}, dy : {dy}, scale : {scale}")
        if scale == 2:
            temp = cv.pyrUp(image_proc)
            disp_temp = cv.pyrUp(disp)
        elif scale == 4:
            temp = cv.pyrUp(cv.pyrUp(image_proc))
            disp_temp = cv.pyrUp(cv.pyrUp(disp))
        else:
            temp = image_proc.copy()
            disp_temp = disp
        mask = disp_temp == k
        temp[mask == 0] = 0
        # print(f"shape : {temp[mask > 0].shape}")
        if left2right:
            if delta == 0:
                new_image[:, dx:][temp > 0] = temp[temp > 0]
            else:
                new_image[:, dx + k:k][temp > 0] = temp[temp > 0]
        else:
            if delta == 0:
                new_image[:, :-dx][temp > 0] = temp[temp > 0]
            else:
                new_image[:, k:-dx + k][temp > 0] = temp[temp > 0]
        # cv.imshow('pixels copied', temp)
        # cv.imshow('mask', np.uint8(mask) * 255)
        # cv.imshow('new image', new_image/255)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    if left2right:
        new_image_cropped = new_image[:, dx:]
    else:
        new_image_cropped = new_image[:, :-dx]
    print(f"    Reconstruction done in : {time.time() - start} seconds")
    '''
    Image post processing options
    '''
    if inpainting:
        start = time.time()
        mask = np.uint8(np.array(((new_image_cropped > 253) + (new_image_cropped < 1)) * 255))
        new_image_cropped = cv.inpaint(np.uint8(new_image_cropped), mask, 10, cv.INPAINT_NS)
        print(f"    Inpainting done in : {time.time() - start} seconds")
    if closing_bool:
        start = time.time()
        if closing_bool > 3 or closing_bool < 0:
            closing_bool = 1
        if len(imageL.shape) == 2:
            footprint = disk(closing_bool*scale * 2 + 1)
        else:
            footprint = ball(closing_bool*scale * 2 + 1)
        new_image_cropped[new_image_cropped == 0] = closing(new_image_cropped, footprint)[
            new_image_cropped == 0]
        print(f"    Closing done in : {time.time() - start} seconds")
    if median:
        start = time.time()
        new_image_cropped[new_image_cropped == 0] = median_filter(new_image_cropped, size=median)[
            new_image_cropped == 0]
        new_image_cropped[new_image_cropped == 0] = median_filter(new_image_cropped, size=median)[
            new_image_cropped == 0]
        print(f"    Median filtering done in : {time.time() - start} seconds")
    if scale == 2:
        new_image_cropped = cv.pyrDown(new_image_cropped)
    elif scale == 4:
        new_image_cropped = cv.pyrDown(cv.pyrDown(new_image_cropped))
    if verbose:
        # print(new_image_cropped.min(), new_image_cropped.max())
        cv.imshow('Original image Left', imageL)
        cv.imshow('Original image Right', imageR)
        cv.imshow('Disparity image', disparity_matrix)
        temp = new_image_cropped / 255 * 0.5
        temp[temp == 0] = image_proc[temp == 0] / 255 * 0.5
        cv.imshow('difference source / Result', temp + image_proc / 255 * 0.5)
        cv.imshow('difference Left Right', imageL / 255 * 0.5 + imageR / 255 * 0.5)
        plt.matshow(disp)
        plt.show()
    return new_image_cropped


def error_estimation(image_translated, ref, ignore_undefined=True):
    if ignore_undefined:
        im1 = image_translated[image_translated > 0]
        im2 = ref[image_translated > 0]
    else:
        im1 = image_translated
        im2 = ref
    ssim_result, grad, s = ssim(im1, im2, data_range=im2.max() - im2.min(), gradient=True, full=True)
    rmse_result = mean_squared_error(im1, im2)
    nmi_result = normalized_mutual_information(im1, im2, bins=100)
    print(f"\n  Structural Similarity Index : {ssim_result}")
    print(f"    Root Mean squared error : {rmse_result}")
    print(f"    Normalized mutual information : {nmi_result}")
    if ignore_undefined:
        print(f"    Percentage of the image defined = {round(len(im1)/(image_translated.shape[0]*image_translated.shape[1])*100,2)}%")
    cv.destroyAllWindows()


def SIFT(image_dst, image_src, MIN_MATCH_COUNT=4, matcher='FLANN', name='00', lowe_ratio=0,
         nfeatures=0, nOctaveLayers=4, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6, toggle_evaluate=False):
    pts_src = {}
    pts_dst = {}
    ## Image in grayscale
    gray_dst = image_dst.GRAYSCALE()
    gray_src = image_src.GRAYSCALE()

    # Initiate SIFT detector
    sift = cv.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold,
                          edgeThreshold=edgeThreshold, sigma=sigma)
    # find the keypoints and descriptors with SIFT
    tic = timeit.default_timer()
    kp_dst, des_dst = sift.detectAndCompute(gray_dst, None)
    kp_src, des_src = sift.detectAndCompute(gray_src, None)
    toc = timeit.default_timer()
    if toggle_evaluate:
        t1 = round(toc - tic, 2)
    else:
        print(
            f"Number of keypoints found in source image : {len(kp_src)}, in the destination image : {len(kp_dst)} in {round(toc - tic, 2)} secondes")
    ## Initialize the matcher and match the keypoints
    tic = timeit.default_timer()
    matcher = init_matcher(method='SIFT', matcher=matcher, trees=5)
    matches = matcher.knnMatch(des_dst, des_src, k=2)
    toc = timeit.default_timer()
    if toggle_evaluate:
        t2 = round(toc - tic, 2)
    else:
        print(f"Matches computed in {round(toc - tic, 2)} secondes")
    # store all the good matches as per Lowe's ratio test.
    good = []
    if lowe_ratio == 0:
        ratio_matches = 0.45
        k = 0.005
        while len(good) < MIN_MATCH_COUNT and ratio_matches < 0.95:
            good = []
            ratio_matches += k
            for m, n in matches:
                if m.distance < ratio_matches * n.distance:
                    good.append(m)
    else:
        ratio_matches = lowe_ratio
        for m, n in matches:
            if m.distance < ratio_matches * n.distance:
                good.append(m)
    if len(good) >= 0:
        src_pts = np.float32([kp_src[m.trainIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp_dst[m.queryIdx].pt for m in good]).reshape(-1, 2)
        #         M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        #         matchesMask = mask.ravel().tolist()
        pts_src[name] = np.int32(src_pts)
        pts_dst[name] = np.int32(dst_pts)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
    if toggle_evaluate:
        return len(kp_src), len(kp_dst), len(src_pts), t1, t2
    else:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)
        img3 = cv.drawMatches(image_dst, kp_dst, image_src, kp_src, good, None, **draw_params)
        plt.figure(figsize=(30, 30), dpi=40)
        plt.imshow(img3, 'gray'), plt.show()
        return pts_src, pts_dst


def init_matcher(method='SIFT', matcher='FLANN', trees=5):
    FLANN_INDEX_LINEAR = 0
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_KMEANS = 2
    FLANN_INDEX_COMPOSITE = 3
    FLANN_INDEX_KDTREE_SINGLE = 4
    FLANN_INDEX_HIERARCHICAL = 5
    FLANN_INDEX_LSH = 6
    FLANN_INDEX_SAVED = 254
    FLANN_INDEX_AUTOTUNED = 255

    if method == 'SIFT':
        norm = cv.NORM_L2
    elif method == 'RootSIFT':
        norm = cv.NORM_L2
    elif method == 'ORB':
        norm = cv.NORM_HAMMING
    elif method == 'BRISK':
        norm = cv.NORM_HAMMING

    if matcher == 'FLANN':
        if norm == cv.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=trees)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12  The number of hash tables to use
                                key_size=6,  # 20  The length of the key in the hash tables
                                multi_probe_level=0)  # 2 Number of levels to use in multi-probe (0 for standard LSH)
        matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    elif matcher == 'BF':
        matcher = cv.BFMatcher(norm)
    else:
        flann_params = dict(algorithm=FLANN_INDEX_LINEAR)
        matcher = cv.FlannBasedMatcher(flann_params, {})
    return matcher
