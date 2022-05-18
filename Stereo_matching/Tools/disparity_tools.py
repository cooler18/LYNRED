import time
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import area_closing, closing, square, cube, disk, ball
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, normalized_mutual_information
from FUSION.classes.Image import ImageCustom


def reconstruction_from_disparity(imageL, imageR, disparity_matrix, min_disp=0, max_disp=0, closing_bool=False,
                                  verbose=False, median=False, inpainting=False, copycat=False, orientation=0):
    """
        Reconstruction function from disparity map.
        :return: reconstruct image.
    """
    start = time.time()
    # Disparity map hold between 0 and 1 with the min/max  disparity pass as arguments.
    if min_disp == max_disp:
        if abs(disparity_matrix).min() >= disparity_matrix.max():
            min_disp = disparity_matrix.max()
            max_disp = -(abs(disparity_matrix).max())
        else:
            min_disp = disparity_matrix.min()
            max_disp = disparity_matrix.max()
        disp = np.round(disparity_matrix.copy())
    else:
        disp = np.round(disparity_matrix * (max_disp - min_disp) + min_disp)
    if not(max_disp == min_disp):
        right2left = max_disp > min_disp
        dx = int(abs(max_disp) * np.cos(orientation))
        dy = int(abs(max_disp) * np.sin(orientation))
        if right2left:
            image_proc = imageL.copy()
        else:
            image_proc = imageR.copy()
        if len(image_proc.shape) > 2:
            new_image = np.zeros([imageL.shape[0] + dy, imageL.shape[1] + dx, 3])
            disp_temp = np.stack([disp, disp, disp], axis=2)
        else:
            new_image = np.zeros([imageL.shape[0] + dy, imageL.shape[1] + dx])
            disp_temp = disp.copy()
        if right2left:
            step = 1
        else:
            step = -1
        for k in range(int(min_disp), int(max_disp), step):
            temp = image_proc.copy()
            mask = disp_temp == k
            temp[mask == 0] = 0
            if right2left:
                if k == 0:
                    new_image[:, dx:][temp > 0] = temp[temp > 0]
                else:
                    new_image[:, dx - k:-k][temp > 0] = temp[temp > 0]
            else:
                if k == dx:
                    new_image[:, :-dx][temp > 0] = temp[temp > 0]
                else:
                    new_image[:, - k:-dx - k][temp > 0] = temp[temp > 0]
            if verbose:
                cv.imshow('New image', new_image / 255)
                cv.waitKey(int(1000 / max_disp))
        if right2left:
            new_image_cropped = new_image[:, dx:]
        else:
            new_image_cropped = new_image[:, :-dx]
    else:
        new_image_cropped = imageL.copy()
    if verbose:
        print(f"    Reconstruction done in : {time.time() - start} seconds")
    '''
    Image post processing options
    '''
    if copycat:
        start = time.time()
        if len(imageR.shape) > len(new_image_cropped.shape):
            copycat = cv.cvtColor(imageR, cv.COLOR_BGR2GRAY)
        elif len(imageR.shape) < len(new_image_cropped.shape):
            copycat = np.stack([imageR, imageR, imageR])
        else:
            copycat = imageR
        mask = ((new_image_cropped > 253) + (new_image_cropped < 1)).astype(np.uint8)
        new_image_cropped[np.where(mask == 1)] = copycat[np.where(mask == 1)]
        if verbose:
            print(f"    Copycat done in : {round(time.time() - start, 2)} seconds")
    elif inpainting:
        start = time.time()
        if len(new_image_cropped.shape) > 2:
            mask = np.uint8(np.array(((cv.cvtColor(new_image_cropped.astype(np.uint8), cv.COLOR_BGR2GRAY) > 253) +
                                  (cv.cvtColor(new_image_cropped.astype(np.uint8), cv.COLOR_BGR2GRAY) < 1)) * 255))
        else:
            mask = np.uint8(np.array(((new_image_cropped > 253) + (new_image_cropped < 1)) * 255))
        new_image_cropped = cv.inpaint(np.uint8(new_image_cropped), mask, 10, cv.INPAINT_NS)
        if verbose:
            print(f"    Inpainting done in : {round(time.time() - start, 2)} seconds")
    elif closing_bool:
        start = time.time()
        if closing_bool > 3 or closing_bool < 0:
            closing_bool = 1
        if len(imageL.shape) == 2:
            footprint = disk(closing_bool * 2 + 1)
        else:
            footprint = ball(closing_bool * 2 + 1)
        new_image_cropped[new_image_cropped == 0] = closing(new_image_cropped, footprint)[
            new_image_cropped == 0]
        if verbose:
            print(f"    Closing done in : {round(time.time() - start, 2)} seconds")
    elif median:
        start = time.time()
        new_image_cropped[new_image_cropped == 0] = median_filter(new_image_cropped, size=median)[
            new_image_cropped == 0]
        new_image_cropped[new_image_cropped == 0] = median_filter(new_image_cropped, size=median)[
            new_image_cropped == 0]
        if verbose:
            print(f"    Median filtering done in : {round(time.time() - start, 2)} seconds")
    if verbose:
        # print(new_image_cropped.min(), new_image_cropped.max())
        cv.imshow('Original image Left', imageL)
        cv.imshow('Original image Right', imageR)
        cv.imshow('Disparity image', disparity_matrix/70)
        temp = new_image_cropped / 255 * 0.5
        temp[temp == 0] = image_proc[temp == 0] / 255 * 0.5
        cv.imshow('difference source / Result', temp + image_proc / 255 * 0.5)
        if len(imageR.shape) == len(new_image_cropped.shape):
            cv.imshow('difference Left Right', imageL / 255 * 0.5 + imageR / 255 * 0.5)
        plt.matshow(disp)
        plt.show()
        cv.destroyAllWindows()
    return new_image_cropped


def reprojection_disparity(disparity_map, translation_ratio, verbose=False):
    """
        Reprojection of a disparity map to another position.
        The disparity is given to project right to left, a negative sign means left to right.
        :return: reconstruct disparity map.
    """
    start = time.time()
    new_values = np.round(disparity_map * (1 - translation_ratio))
    new_ref = np.round(disparity_map * translation_ratio)
    if new_ref.max() <= 0:
        max_disp = -abs(new_ref).max()
        min_disp = -abs(new_ref).min()
    else:
        max_disp = new_ref.max()
        min_disp = new_ref.min()
    right2left = max_disp > min_disp
    if max_disp * min_disp < 0:
        dx = int(abs(max_disp - min_disp))
    else:
        dx = int(abs(max_disp))
    new_disparity = np.ones([disparity_map.shape[0], disparity_map.shape[1] + int(dx)]) * -1000
    if right2left:
        step = 1
    else:
        step = -1
    for k in range(int(min_disp), int(max_disp), step):
        mask = new_ref == k
        if right2left:
            if k == 0:
                new_disparity[:, dx:][mask == 1] = new_values[mask == 1]
            else:
                new_disparity[:, dx - k:-k][mask == 1] = new_values[mask == 1]
        else:
            if k == dx:
                new_disparity[:, :-dx][mask == 1] = new_values[mask == 1]
            else:
                new_disparity[:, -k:-dx - k][mask == 1] = new_values[mask == 1]
        if verbose:
            temp = new_disparity.copy()
            temp[temp == -1000] = 0
            if temp.max() <= 0:
                m = -(abs(temp).max())
            else:
                m = temp.max()
            cv.imshow('New disparity image', temp / m)
            cv.waitKey(int(abs(1000 / max_disp)))
    if not right2left:
        new_disparity_cropped = new_disparity[:, dx:]
    else:
        new_disparity_cropped = new_disparity[:, :-dx]
    if verbose:
        print(f"    Reconstruction done in : {time.time() - start} seconds")
    '''
    Image post processing options
    '''
    start = time.time()
    new_disparity_cropped[new_disparity_cropped == -1000] = 0
    new_disparity_cropped[new_disparity_cropped == 0] = median_filter(new_disparity_cropped, size=5)[
        new_disparity_cropped == 0]
    if verbose:
        print(f"    Median filtering done in : {round(time.time() - start, 2)} seconds")
        if new_disparity_cropped.max() <= 0:
            m = -abs(new_disparity_cropped).max()
        cv.imshow('Original disparity image', disparity_map / disparity_map.max())
        cv.imshow('New disparity image', new_disparity_cropped / m)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return new_disparity_cropped


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


def disparity_post_process(disparity_matrix, min_disp, max_disp, threshold, verbose=False):
    start = time.time()
    if min_disp == 0 and max_disp == 0:
        disp = disparity_matrix
    else:
        disp = np.round(disparity_matrix * (max_disp - min_disp) + min_disp)
        if threshold == 0:
            threshold = 70
            mask_false_values = ((disp > threshold) + (disp < 2 + min_disp))
            disp[mask_false_values] = 0
        else:
            mask_false_values = ((disp > threshold) + (disp < 2 + min_disp))
            good_values = (disp * (mask_false_values == 0))
            sum_rows = good_values.sum(1)
            nb_values_sum = (good_values != 0).sum(1)
            nb_values_sum[nb_values_sum == 0] = 1
            ref = sum_rows / nb_values_sum
            ref = np.expand_dims(ref, axis=1) * np.ones([1, disparity_matrix.shape[1]])
            disp[mask_false_values] = ref[mask_false_values]

    if verbose:
        print(f"    Post-processing of the disparity map done in {round(time.time() - start, 2)} seconds")
    if min_disp == 0 and max_disp == 0:
        return disp
    else:
        return (disp - min_disp) / (max_disp - min_disp)
