import os
import time
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy.ndimage import median_filter


from Stereo_matching.Tools.dataloader import data_superloader


def depthMapping(sample, source, min_disp, max_disp, binary, verbose, time_of_day='Day', im_type='visible', threshold=0):
    """
        main function applying the semi-global matching algorithm of OpenCv.
        :return: imgL, imgR, disparity_n, m, M.
    """
    """
        Tool functions
    """
    def simple_inference(imgL, imgR, stereo):
        disparity = stereo.compute(imgL, imgR)
        # Converting to float32
        disparity = disparity.astype(np.float32)
        # Scaling down the disparity values and normalizing them
        disparity = disparity / 16
        if disparity.min() < 0 and disparity.max() > 0:
            disparity[disparity < 0] = 0
        disparity[disparity == disparity.max()] = median_filter(disparity, size=5)[disparity == disparity.max()]
        return disparity

    @data_superloader(time_of_day, im_type, post_process=threshold, clean=False, path_save=None)
    def multiple_inference(sample, stereo):
        imgL, imgR = sample['imgL'], sample['imgR']
        disparity = stereo.compute(imgL, imgR)
        # Converting to float32
        disparity = disparity.astype(np.float32)
        # Scaling down the disparity values and normalizing them
        disparity = disparity / 16
        if disparity.min() < 0 and disparity.max() > 0:
            disparity[disparity < 0] = 0
        disparity[disparity == disparity.max()] = median_filter(disparity, size=5)[disparity == disparity.max()]
        return disparity

    def nothing(x):
        pass

    if not(sample is None):
        imgL = np.uint8(sample['imgL']*255)
        imgR = np.uint8(sample['imgR']*255)
    if source == 'teddy':
        numDisparities = 2
        blockSize = 1
        preFilterType = 1
        preFilterSize = 1
        preFilterCap = 62
        textureThreshold = 100
        uniquenessRatio = 10
        speckleRange = 100
        speckleWindowSize = 50
        disp12MaxDiff = 50
        minDisparity = 20
    elif source == 'cones':
        numDisparities = 2
        blockSize = 1
        preFilterType = 1
        preFilterSize = 1
        preFilterCap = 62
        textureThreshold = 100
        uniquenessRatio = 10
        speckleRange = 100
        speckleWindowSize = 50
        disp12MaxDiff = 50
        minDisparity = 20
    elif source == "lynred":
        imgL = cv.pyrDown(imgL)
        imgR = cv.pyrDown(imgR)
        numDisparities = 11
        blockSize = 0
        preFilterType = 1
        preFilterSize = 25
        preFilterCap = 0
        textureThreshold = 100
        uniquenessRatio = 10
        speckleRange = 100
        speckleWindowSize = 50
        disp12MaxDiff = 150
        minDisparity = 20
    elif source == 'lynred_inf':
        numDisparities = 11
        blockSize = 0
        preFilterType = 1
        preFilterSize = 25
        preFilterCap = 0
        textureThreshold = 100
        uniquenessRatio = 0
        speckleRange = 5
        speckleWindowSize = 50
        disp12MaxDiff = 150
        minDisparity = 85
    elif source == 'lynred_vis':
        numDisparities = 11
        blockSize = 0
        preFilterType = 1
        preFilterSize = 25
        preFilterCap = 0
        textureThreshold = 100
        uniquenessRatio = 0
        speckleRange = 5
        speckleWindowSize = 50
        disp12MaxDiff = 150
        minDisparity = 85
    else:
        raise ValueError("Invalid source name")
    P1, P2 = 50, 1000

    if verbose:
        cv.namedWindow('disp', cv.WINDOW_NORMAL)
        cv.resizeWindow('disp', 200, 600)
        cv.createTrackbar('numDisparities', 'disp', numDisparities, int(max_disp / 16), nothing)
        cv.createTrackbar('minDisparity', 'disp', minDisparity, 150, nothing)
        cv.createTrackbar('blockSize', 'disp', blockSize, 50, nothing)
        cv.createTrackbar('preFilterCap', 'disp', preFilterCap, 62, nothing)
        cv.createTrackbar('uniquenessRatio', 'disp', uniquenessRatio, 100, nothing)
        cv.createTrackbar('speckleRange', 'disp', speckleRange, 100, nothing)
        cv.createTrackbar('speckleWindowSize', 'disp', speckleWindowSize, 50, nothing)
        cv.createTrackbar('disp12MaxDiff', 'disp', disp12MaxDiff, 150, nothing)
        if not binary:
            cv.createTrackbar('preFilterType', 'disp', preFilterType, 1, nothing)
            cv.createTrackbar('textureThreshold', 'disp', textureThreshold, 100, nothing)
            cv.createTrackbar('preFilterSize', 'disp', preFilterSize, 25, nothing)
            stereo = cv.StereoBM_create()
            if len(imgL.shape) > 2:
                imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
            if len(imgR.shape) > 2:
                imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        else:
            cv.createTrackbar('P1', 'disp', P1, 500, nothing)
            cv.createTrackbar('P2', 'disp', P2, 1000, nothing)
            stereo = cv.StereoSGBM_create()
            stereo.setMode(1)

        cv.resizeWindow('disp', 3 * imgL.shape[1], int(imgL.shape[0] * 1.5))
        while True:
            # Updating the parameters based on the trackbar positions
            numDisparities = (cv.getTrackbarPos('numDisparities', 'disp') + 1) * 16
            blockSize = cv.getTrackbarPos('blockSize', 'disp') * 2 + 5
            preFilterCap = cv.getTrackbarPos('preFilterCap', 'disp') + 1
            uniquenessRatio = cv.getTrackbarPos('uniquenessRatio', 'disp')
            speckleRange = cv.getTrackbarPos('speckleRange', 'disp')
            speckleWindowSize = cv.getTrackbarPos('speckleWindowSize', 'disp') * 2
            disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff', 'disp')
            minDisparity = int(-cv.getTrackbarPos('minDisparity', 'disp') + min_disp)
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
            start = time.time()
            disparity = stereo.compute(imgL, imgR)
            # NOTE: Code returns a 16bit signed single channel image,
            # CV_16S containing a disparity map scaled by 16. Hence it
            # is essential to convert it to CV_32F and scale it down 16 times.
            # Converting to float32
            disparity = disparity.astype(np.float32)

            # Scaling down the disparity values and normalizing them
            disparity = disparity / 16
            if disparity.max() < 0:
                m = disparity.max()
                M = disparity.min()
            else:
                if disparity.min() < 0:
                    disparity[disparity < 0] = 0
                m = disparity.min()
                M = disparity.max()

            if M < 0:
                disparity_n = (disparity - m) / (M - m)
            else:
                disparity_n = (disparity - m) / (M - m)

            disparity_n[disparity_n == 1] = median_filter(disparity_n, size=5)[disparity_n == 1]

            print(f"    Done in : {time.time() - start} secondes")
            # if not source == 'drive':
            #     disparity = cv.bitwise_not(disparity)
            if binary:
                final = np.hstack([cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)/255, disparity_n, cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)/255])
            else:
                final = np.hstack([imgR/255, disparity_n, imgL/255])
            # Displaying the disparity map
            if verbose:
                cv.imshow("disp", final)
            if cv.waitKey(1) == 27 or not verbose:
                break
        cv.destroyAllWindows()
        print(f"Max disparity = {M}\n"
              f"Min disparity = {m}")
        plt.matshow(disparity)
        plt.show()
        return imgL, imgR, disparity_n, m, M
    else:
        # Updating the parameters based on the trackbar positions
        numDisparities = (numDisparities + 1) * 16
        blockSize = blockSize * 2 + 5
        preFilterCap = preFilterCap + 1
        speckleWindowSize = speckleWindowSize * 2
        minDisparity = int(-minDisparity + min_disp)
        if binary:
            P2 = max(P1 + 1, P2)
            stereo = cv.StereoSGBM_create()
            stereo.setMode(1)
            stereo.setP1(P1)
            stereo.setP2(P2)
        else:
            preFilterSize = preFilterSize * 2 + 5
            stereo = cv.StereoBM_create()
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
        start = time.time()
        if not(sample is None):
            disparity_matrix = simple_inference(imgL, imgR, stereo)
            m, M = disparity_matrix.min(), disparity_matrix.max()
            return imgL, imgR, disparity_matrix, m, M
        else:
            multiple_inference(None, stereo)
