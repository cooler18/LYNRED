import argparse
import os
import time
from os.path import join
import numpy as np
import cv2 as cv
from cv2 import dilate
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter, prewitt

from FUSION.tools.registration_tools import reconstruction_from_disparity


def grad(image):
    Ix = cv.Sobel(image, cv.CV_64F, 1, 0, borderType=cv.BORDER_REFLECT_101)
    Iy = cv.Sobel(image, cv.CV_64F, 0, 1, borderType=cv.BORDER_REFLECT_101)
    return np.uint8(np.sqrt(Ix ** 2 + Iy ** 2)/2.1+image/2.1)


def nothing(x):
    pass


def pre_proc(image, typ):
    return image
    # if typ == 'ir':
    #     # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     # return clahe.apply(image)
    #     return grad(cv.bitwise_not(image))
    # if typ == 'vis':
    #     return grad(image)


def depthMapping(source, min_disp, binary, verbose, edges):
    """
        main function applying the semi-global matching algorithm of OpenCv.
        :return: imgL, imgR, disparity_n, m, M.
    """
    if source == 'drive_vis' or source == 'drive_inf':
        fold = np.random.randint(0, 1)
        if fold:
            p_drive = 'D:\Travail\LYNRED\Day'
        else:
            p_drive = 'D:\Travail\LYNRED/Night'
        p_drive_left = join(p_drive, 'slave', "visible")
        p_drive_right = join(p_drive, 'master', "visible")
        n = np.random.randint(0, len(os.listdir(p_drive_left)) - 1)
        l_drive = join(p_drive_left, os.listdir(p_drive_left)[n])
        r_drive = join(p_drive_right, os.listdir(p_drive_right)[n])

    p = 'D:\Travail\LYNRED\Stereo_matching\Samples/'
    l_cones = join(p, 'left_cones.png')
    r_cones = join(p, 'right_cones.png')
    l_teddy = join(p, 'left_teddy.png')
    r_teddy = join(p, 'right_teddy.png')

    if source == 'teddy':
        imgL = cv.imread(l_teddy, int(binary))
        imgR = cv.imread(r_teddy, int(binary))
        numDisparities = 1
        blockSize = 1
        preFilterType = 1
        preFilterSize = 1
        preFilterCap = 62
        textureThreshold = 0
        uniquenessRatio = 1
        speckleRange = 100
        speckleWindowSize = 50
        disp12MaxDiff = 50
        minDisparity = 50
    elif source == 'cones':
        imgL = cv.imread(l_cones, int(binary))
        imgR = cv.imread(r_cones, int(binary))
        numDisparities = 2
        blockSize = 1
        preFilterType = 1
        preFilterSize = 1
        preFilterCap = 17
        textureThreshold = 0
        uniquenessRatio = 1
        speckleRange = 100
        speckleWindowSize = 50
        disp12MaxDiff = 50
        minDisparity = 70
    elif source == "drive_vis" or source == 'drive_inf':
        imgL = pre_proc(cv.imread(l_drive, int(binary)), 'ir')
        imgR = pre_proc(cv.imread(r_drive, int(binary)), 'vis')
        imgL = cv.pyrDown(imgL)
        imgR = cv.pyrDown(imgR)
        numDisparities = 3
        blockSize = 0
        preFilterType = 1
        preFilterSize = 25
        preFilterCap = 0
        textureThreshold = 100
        uniquenessRatio = 10
        speckleRange = 100
        speckleWindowSize = 50
        disp12MaxDiff = 150
        minDisparity = 90
    else:
        raise ValueError("Invalid source name")
    if binary:
        P1, P2 = 0, 1000

    cv.namedWindow('disp', cv.WINDOW_NORMAL)
    cv.resizeWindow('disp', 200, 600)
    cv.createTrackbar('numDisparities', 'disp', numDisparities, 16, nothing)
    cv.createTrackbar('minDisparity', 'disp', minDisparity, 150, nothing)
    cv.createTrackbar('blockSize', 'disp', blockSize, 50, nothing)
    # print(binary)
    if not binary:
        cv.createTrackbar('preFilterType', 'disp', preFilterType, 1, nothing)
        cv.createTrackbar('textureThreshold', 'disp', textureThreshold, 100, nothing)
        cv.createTrackbar('preFilterSize', 'disp', preFilterSize, 25, nothing)
    else:
        cv.createTrackbar('P1', 'disp', P1, 500, nothing)
        cv.createTrackbar('P2', 'disp', P2, 1000, nothing)

    cv.createTrackbar('preFilterCap', 'disp', preFilterCap, 62, nothing)
    cv.createTrackbar('uniquenessRatio', 'disp', uniquenessRatio, 100, nothing)
    cv.createTrackbar('speckleRange', 'disp', speckleRange, 100, nothing)
    cv.createTrackbar('speckleWindowSize', 'disp', speckleWindowSize, 50, nothing)
    cv.createTrackbar('disp12MaxDiff', 'disp', disp12MaxDiff, 150, nothing)

    cv.resizeWindow('disp', 1800, 600)
    # Creating an object of StereoBM algorithm
    if binary:
        stereo = cv.StereoSGBM_create()
        stereo.setMode(1)
    else:
        stereo = cv.StereoBM_create()
    imgR_gray = imgL
    imgL_gray = imgR

    # Applying stereo image rectification on the left image
    # Left_nice = cv.remap(imgL_gray, Left_Stereo_Map_x, Left_Stereo_Map_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    #
    # # Applying stereo image rectification on the right image
    # Right_nice = cv.remap(imgR_gray, Left_Stereo_Map_x, Left_Stereo_Map_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    while True:
        # Updating the parameters based on the trackbar positions
        numDisparities = (cv.getTrackbarPos('numDisparities', 'disp') + 1) * 16
        blockSize = cv.getTrackbarPos('blockSize', 'disp') * 2 + 5
        preFilterCap = cv.getTrackbarPos('preFilterCap', 'disp') + 1
        uniquenessRatio = cv.getTrackbarPos('uniquenessRatio', 'disp')
        speckleRange = cv.getTrackbarPos('speckleRange', 'disp')
        speckleWindowSize = cv.getTrackbarPos('speckleWindowSize', 'disp') * 2
        disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff', 'disp')
        minDisparity = int(-cv.getTrackbarPos('minDisparity', 'disp')/2 - min_disp)
        if binary:
            P1 = cv.getTrackbarPos('P1', 'disp')
            P2 = max(P1+1, cv.getTrackbarPos('P2', 'disp'))
        else:
            preFilterType = cv.getTrackbarPos('preFilterType', 'disp')
            preFilterSize = cv.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
            textureThreshold = cv.getTrackbarPos('textureThreshold', 'disp')
        # Setting the updated parameters before computing disparity map
        if binary:
            stereo.setP1(P1)
            stereo.setP2(P2)
        else:
            stereo.setTextureThreshold(textureThreshold)
            stereo.setPreFilterType(preFilterType)
            stereo.setPreFilterSize(preFilterSize)
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # Calculating disparity using the StereoBM algorithm

        if source == 'test':
            dis = np.zeros([imgL.shape[0], imgL.shape[1], 3])
            dis[:, :, 0] = stereo.compute(imgL_gray[:, :, 0], imgR_gray[:, :, 0])
            dis[:, :, 1] = stereo.compute(imgL_gray[:, :, 1], imgR_gray[:, :, 1])
            dis[:, :, 2] = stereo.compute(imgL_gray[:, :, 2], imgR_gray[:, :, 2])
            disparity = dis.mean(axis=2)
        else:
            start = time.time()
            disparity = stereo.compute(imgL_gray, imgR_gray)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it
        # is essential to convert it to CV_32F and scale it down 16 times.
        # Converting to float32
        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them
        disparity = disparity / 16
        if disparity.min() < 0:
            m = -abs(disparity).min()
            M = -abs(disparity).max()
        else:
            m = disparity.min()
            M = disparity.max()

        # print(m, M)
        if len(imgL_gray.shape) > 2:
            im = cv.cvtColor(imgL_gray, cv.COLOR_BGR2GRAY)/255
        else:
            im = imgL_gray/255
        Ix = prewitt(im, axis=0)
        Iy = prewitt(im, axis=1)
        grad_L = np.sqrt(Ix ** 2 + Iy ** 2)
        th = 0.1
        _, grad_L = cv.threshold(grad_L/grad_L.max()*255, 255*th, 255, cv.THRESH_BINARY)
        if M < 0:
            disparity_n = (disparity - m) / (M - m)
        else:
            disparity_n = (disparity - m) / (M - m)
        disparity_n[disparity_n == 1] = median_filter(disparity_n, size=5)[disparity_n == 1]
        # disparity_n[disparity_n == 1] = 0
        # disparity_n[grad_L < th] = 0
        # print(m, M)

        print(f"    Done in : {time.time() - start} secondes")
        # if not source == 'drive':
        #     disparity = cv.bitwise_not(disparity)
        if edges:
            kernel = np.ones((3, 3), np.uint8)
            grad_L = dilate(grad_L, kernel, iterations=1)
            disparity_n[grad_L > 0] = 1
        if binary:
            final = np.hstack([cv.cvtColor(imgR_gray, cv.COLOR_BGR2GRAY)/255, disparity_n, im])
        else:
            final = np.hstack([imgR_gray/255, disparity_n, imgL_gray/255])
        # Displaying the disparity map
        if verbose:
            cv.imshow("disp", final)
        if cv.waitKey(1) == 27 or not verbose:
            break

    cv.destroyAllWindows()
    return imgL, imgR, disparity_n, m, M

# if __name__ == '__main__':
#     depthMapping()