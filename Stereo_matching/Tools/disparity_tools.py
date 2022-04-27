import time
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import area_closing, closing, square, cube, disk, ball
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, normalized_mutual_information


def reconstruction_from_disparity(imageL, imageR, disparity_matrix, min_disp, max_disp, scale, closing_bool, verbose,
                                  median, inpainting, copycat, orientation=0):
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

    left2right = max_disp - min_disp > 0
    disp = np.round(disparity_matrix * (max_disp - min_disp) * scale + min_disp * scale)
    dx = int(abs(max_disp) * scale * np.cos(orientation))
    dy = int(abs(max_disp) * scale * np.sin(orientation))
    if left2right:
        # if len(imageL.shape) > 2:
        #     image_proc = cv.cvtColor(imageL, cv.COLOR_BGR2GRAY)
        # else:
        image_proc = imageL.copy()
    else:
        # if len(imageR.shape) > 2:
        #     image_proc = cv.cvtColor(imageR, cv.COLOR_BGR2GRAY)
        # else:
        image_proc = imageR.copy()
    if len(imageR.shape) > 2:
        new_image = np.zeros([imageL.shape[0] * scale + dy, imageL.shape[1] * scale + dx, 3])
        disp_temp = np.stack([disp, disp, disp], axis=2)
    else:
        new_image = np.zeros([imageL.shape[0] * scale + dy, imageL.shape[1] * scale + dx])
        disp_temp = disp.copy()
    if left2right:
        step = 1
    else:
        step = -1
    for k in range(int(min_disp * scale), int(max_disp * scale), step):
        delta = k - round(min_disp * scale)
        if scale == 2:
            temp = cv.pyrUp(image_proc)
            disp_temp = cv.pyrUp(disp_temp)
        elif scale == 4:
            temp = cv.pyrUp(cv.pyrUp(image_proc))
            disp_temp = cv.pyrUp(cv.pyrUp(disp_temp))
        else:
            temp = image_proc.copy()
        mask = disp_temp == k
        temp[mask == 0] = 0
        if left2right:
            if k == 0:
                new_image[:, dx:][temp > 0] = temp[temp > 0]
            else:
                new_image[:, dx - k:-k][temp > 0] = temp[temp > 0]
        else:
            if k == 0:
                new_image[:, :-dx][temp > 0] = temp[temp > 0]
            else:
                new_image[:, k:-dx + k][temp > 0] = temp[temp > 0]
        if verbose:
            cv.imshow('new image', new_image / 255)
            cv.waitKey(int(1000 / max_disp))
    if left2right:
        new_image_cropped = new_image[:, dx:]
    else:
        new_image_cropped = new_image[:, :-dx]
    print(f"    Reconstruction done in : {time.time() - start} seconds")
    '''
    Image post processing options
    '''
    if copycat:
        start = time.time()
        mask = ((new_image_cropped > 253) + (new_image_cropped < 1)).astype(np.uint8)
        new_image_cropped[np.where(mask == 1)] = imageR[np.where(mask == 1)]
        print(f"    Copycat done in : {round(time.time() - start,2)} seconds")
    elif inpainting:
        start = time.time()
        mask = np.uint8(np.array(((new_image_cropped > 253) + (new_image_cropped < 1)) * 255))
        new_image_cropped = cv.inpaint(np.uint8(new_image_cropped), mask, 10, cv.INPAINT_NS)
        print(f"    Inpainting done in : {round(time.time() - start,2)} seconds")
    elif closing_bool:
        start = time.time()
        if closing_bool > 3 or closing_bool < 0:
            closing_bool = 1
        if len(imageL.shape) == 2:
            footprint = disk(closing_bool * scale * 2 + 1)
        else:
            footprint = ball(closing_bool * scale * 2 + 1)
        new_image_cropped[new_image_cropped == 0] = closing(new_image_cropped, footprint)[
            new_image_cropped == 0]
        print(f"    Closing done in : {round(time.time() - start,2)} seconds")
    elif median:
        start = time.time()
        new_image_cropped[new_image_cropped == 0] = median_filter(new_image_cropped, size=median)[
            new_image_cropped == 0]
        new_image_cropped[new_image_cropped == 0] = median_filter(new_image_cropped, size=median)[
            new_image_cropped == 0]
        print(f"    Median filtering done in : {round(time.time() - start,2)} seconds")
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
    print(f"    Structural Similarity Index : {ssim_result}")
    print(f"    Root Mean squared error: {rmse_result}")
    print(f"    Normalized mutual information : {nmi_result}")
    if ignore_undefined:
        print(
            f"    Percentage of the image defined = {round(len(im1) / (image_translated.shape[0] * image_translated.shape[1]) * 100, 2)}%")
    cv.destroyAllWindows()


def disparity_post_process(disparity_matrix, min_disp, max_disp, threshold):
    start = time.time()
    disp = np.round(disparity_matrix * (max_disp - min_disp) + min_disp)
    mask = disp > threshold
    mask = (disp * (mask == 0))
    ref = np.true_divide(mask.sum(1), (mask != 0).sum(1))
    ref = np.expand_dims(ref, axis=1) * np.ones([1, disparity_matrix.shape[1]])
    disp[disp > threshold] = ref[disp > threshold]
    print(f"    Post-processing of the disparity map done in {round(time.time() - start,2)} seconds")
    return (disp - min_disp) / (max_disp - min_disp)
