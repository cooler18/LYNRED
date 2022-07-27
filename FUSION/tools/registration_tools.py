import os
import pathlib
import pickle
from os.path import join
import cv2 as cv
import timeit
import matplotlib.pyplot as plt
import numpy as np

from FUSION.classes.Image import ImageCustom
from FUSION.tools.gradient_tools import grad


def automatic_registration(imL, imR, matrix, path_save, nameL ="left_rect.png", nameR="right_rect.png", save=True):
    warp_matrix_rotation = matrix["homography"]
    CutY, CutZ = matrix["CutY"], matrix["CutZ"]
    m, n = imL.shape[:2]
    imL_aligned = cv.warpPerspective(imL.copy(), warp_matrix_rotation, (n, m),
                                     flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
    imR_aligned = imR.copy()
    if imR_aligned.shape[0] > imL.shape[0]:
        f = 2
    else:
        f = 1
    if CutZ >= 0:
        if CutY >= 0:
            imR_aligned, imL_aligned = imR_aligned[CutZ*f:, CutY*2:], imL_aligned[CutZ:, CutY:]

        else:
            imR_aligned, imL_aligned = imR_aligned[CutZ*f:, :CutY*f], imL_aligned[CutZ:, :CutY]
    else:
        if CutY >= 0:
            imR_aligned, imL_aligned = imR_aligned[:CutZ*f, CutY*f:], imL_aligned[:CutZ, CutY:]
        else:
            imR_aligned, imL_aligned = imR_aligned[:CutZ*f, :CutY*f], imL_aligned[:CutZ, :CutY]
    if save:
        # if os.path.basename(os.path.dirname(os.path.dirname(path_save))) == "Video_frame":
        #     if imL_aligned.shape[1] > 640:
        #         imL_aligned = cv.pyrDown(imL_aligned)
        #     if imR_aligned.shape[1] > 640:
        #         imR_aligned = cv.pyrDown(imR_aligned)
        if len(imL_aligned.shape) > 2:
            cv.imwrite(path_save + '/left/' + nameL, cv.cvtColor(imL_aligned, cv.COLOR_RGB2BGR))
        else:
            cv.imwrite(path_save + '/left/' + nameL, imL_aligned)

        if len(imR_aligned.shape) > 2:
            cv.imwrite(path_save + '/right/' + nameR, cv.cvtColor(imR_aligned, cv.COLOR_RGB2BGR))
        else:
            cv.imwrite(path_save + '/right/' + nameR, imR_aligned)
    else:
        return imL_aligned, imR_aligned


def manual_registration(imL, imR, path_save, auto=False, verbose=False, delta=None):
    from Stereo_matching.NeuralNetwork.ACVNet_main.ACVNet_main import test_sample
    from Stereo_matching.NeuralNetwork.ACVNet_main.ACVNet_test import initialize_model
    model = initialize_model(verbose=False)

    def nothing(x):
        pass
    m, n = imL.shape[:2]
    if auto:
        if delta is None:
            delta = [0, 200, 0, 5]
        # for i in range(6):
        #     delta = [5 + 20*i, 20+36*i, 0, 5]
            # if imR.shape[0] < 500:
            #     delta = [0 + 10 * i, 10 + 18 * i, 0, 5]
        pts_src, pts_dst = SIFT(imR, imL, MIN_MATCH_COUNT=4, matcher='FLANN', name='00', lowe_ratio=0.5,
                                nfeatures=0, nOctaveLayers=4, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6,
                                toggle_evaluate=False, verbose=verbose, delta=delta)
        warp_matrix_rotation, _ = cv.findHomography(pts_dst, pts_src)
        CutY = -int(warp_matrix_rotation[0, 2])
        CutZ = -int(warp_matrix_rotation[1, 2])
        imL_aligned = cv.warpPerspective(imL.copy(), warp_matrix_rotation, (n, m),
                                         flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
        imR_aligned = imR.copy()
        if CutZ >= 0:
            if CutY >= 0:
                imR_aligned, imL_aligned = imR_aligned[CutZ:, CutY:], imL_aligned[CutZ:, CutY:]

            else:
                imR_aligned, imL_aligned = imR_aligned[CutZ:, :CutY], imL_aligned[CutZ:, :CutY]
        else:
            if CutY >= 0:
                imR_aligned, imL_aligned = imR_aligned[:CutZ, CutY:], imL_aligned[:CutZ, CutY:]
            else:
                imR_aligned, imL_aligned = imR_aligned[:CutZ, :CutY], imL_aligned[:CutZ, :CutY]
        sample = {
            'imgL': cv.pyrDown(imL_aligned) / 255,
            'imgR': cv.pyrDown(imR_aligned) / 255
        }
        image_outputs = test_sample(sample, model)
        disparity = image_outputs["disp_est"]
        if m > 480:
            disparity = disparity / 140
        else:
            disparity = disparity / 70
        final = np.uint8((imR_aligned / 2 + imL_aligned / 2))
        final = cv.resize(final, (disparity.shape[1], disparity.shape[0]))
        final = np.hstack(
            [cv.cvtColor(final, cv.COLOR_RGB2BGR) / 255, np.stack([disparity, disparity, disparity], axis=2)])
        if verbose:
            cv.imshow("Fusion", final)
            cv.waitKey(0)
            cv.destroyAllWindows()
        if len(imL.shape) > 2:
            cv.imwrite(path_save + '/left_rect.png', cv.cvtColor(imL_aligned, cv.COLOR_RGB2BGR))
        else:
            cv.imwrite(path_save + '/left_rect.png', imL_aligned)
        if len(imR.shape) > 2:
            cv.imwrite(path_save + '/right_rect.png', cv.cvtColor(imR_aligned, cv.COLOR_RGB2BGR))
        else:
            cv.imwrite(path_save + '/right_rect.png', cv.cvtColor(imR_aligned, cv.COLOR_RGB2BGR))
        with open(join(path_save, "transform_matrix_auto"), "wb") as p:
            pickle.dump({"homography": warp_matrix_rotation,
                         "rotation": [0, 0, 0],
                         "CutY": CutY,
                         "CutZ": CutZ}, p)
    else:
        try:
            with open(join(path_save, "transform_matrix"), "rb") as p:
                matrix = pickle.load(p)
                print(matrix)
            warp_matrix_rotation = matrix["homography"]
            Rx, Rz, Ry = matrix["rotation"]
            Tx, Ty, Tz = int((warp_matrix_rotation[0, 0] - 0.5) * 500), warp_matrix_rotation[0, 2], \
                         warp_matrix_rotation[1, 2]
            CutY = matrix["CutY"]
            CutZ = matrix["CutZ"]
            # warp_matrix_rotation = matrix["matrix_rotation"]
            # warp_matrix_translation = matrix["matrix_translation"]
            # warp_matrix_rotation = cv.Rodrigues(warp_matrix_rotation)
            # Rx, Rz, Ry = warp_matrix_rotation[0]
            # Tx, Ty, Tz = int((warp_matrix_translation[0, 0] - 0.5) * 500), warp_matrix_translation[0, 2], \
            #              warp_matrix_translation[1, 2]
            with open(join(path_save, "transform_matrix_old"), "wb") as p:
                pickle.dump(matrix, p)
        except FileNotFoundError:
            try:
                with open(join(path_save, "transform_matrix_auto"), "rb") as p:
                    matrix = pickle.load(p)
                warp_matrix_rotation = matrix["homography"]
                Rx, Rz, Ry = matrix["rotation"]
                Tx, Ty, Tz = int((warp_matrix_rotation[0, 0] - 0.5) * 500), warp_matrix_rotation[0, 2], \
                             warp_matrix_rotation[1, 2]
                CutY = matrix["CutY"]
                CutZ = matrix["CutZ"]
            except FileNotFoundError:
                print('No disparity matrix found')
                Rx, Rz, Ry = 0, 0, 0
                Tx, Ty, Tz = 250, 0, 0
                CutY = 0
                CutZ = 0
        cv.namedWindow('Fusion', cv.WINDOW_NORMAL)
        cv.resizeWindow('Fusion', n, m)
        cv.createTrackbar('Rotation Z', 'Fusion', int(Rz*10**6 + 250), 500, nothing)
        cv.createTrackbar('Rotation X', 'Fusion', int(Rx*10**6 + 250), 500, nothing)
        cv.createTrackbar('Rotation Y', 'Fusion', int(Ry*10**4 + 250), 500, nothing)
        cv.createTrackbar('Translation Z', 'Fusion', int(Tz + 250), 500, nothing)
        cv.createTrackbar('Translation Y', 'Fusion',  int(Ty + 250), 500, nothing)
        cv.createTrackbar('Translation X', 'Fusion', Tx, 500, nothing)
        cv.createTrackbar('Cut Z', 'Fusion', CutZ + 250, 500, nothing)
        cv.createTrackbar('Cut Y', 'Fusion', CutY + 250, 500, nothing)
        while True:
            # Updating the parameters based on the trackbar positions
            Rz = (cv.getTrackbarPos('Rotation Z', 'Fusion') - 250)/10**6
            Rx = (cv.getTrackbarPos('Rotation X', 'Fusion') - 250)/10**6
            Ry = (cv.getTrackbarPos('Rotation Y', 'Fusion') - 250)/10**4
            Tz = cv.getTrackbarPos('Translation Z', 'Fusion') - 250
            Ty = cv.getTrackbarPos('Translation Y', 'Fusion') - 250
            Tx = cv.getTrackbarPos('Translation X', 'Fusion')/500 + 0.5
            CutY = cv.getTrackbarPos('Cut Y', 'Fusion') - 250
            CutZ = cv.getTrackbarPos('Cut Z', 'Fusion') - 250
            warp_matrix_rotation, _ = cv.Rodrigues(np.array([Rx, Rz, Ry]))
            warp_matrix_rotation[0, 0] = Tx
            warp_matrix_rotation[1, 1] = Tx
            warp_matrix_rotation[0, 2] = Ty
            warp_matrix_rotation[1, 2] = Tz
            imL_aligned = cv.warpPerspective(imL.copy(), warp_matrix_rotation, (n, m), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
            imR_aligned = imR.copy()
            if CutZ >= 0:
                if CutY >= 0:
                    imR_aligned, imL_aligned = imR_aligned[CutZ:, CutY:], imL_aligned[CutZ:, CutY:]

                else:
                    imR_aligned, imL_aligned = imR_aligned[CutZ:, :CutY], imL_aligned[CutZ:, :CutY]
            else:
                if CutY >= 0:
                    imR_aligned, imL_aligned = imR_aligned[:CutZ, CutY:], imL_aligned[:CutZ, CutY:]
                else:
                    imR_aligned, imL_aligned = imR_aligned[:CutZ, :CutY], imL_aligned[:CutZ, :CutY]
            sample = {
                'imgL': imL_aligned/255,
                'imgR': imR_aligned/255
            }
            image_outputs = test_sample(sample, model)
            disparity = image_outputs["disp_est"]
            if m > 480:
                disparity = disparity / 140
            else:
                disparity = disparity / 70
            final = np.uint8((imR_aligned / 2 + imL_aligned / 2))
            final = cv.resize(final, (disparity.shape[1], disparity.shape[0]))
            final = np.hstack([cv.cvtColor(final, cv.COLOR_RGB2BGR) / 255, np.stack([disparity, disparity, disparity], axis=2)])
            cv.imshow("Fusion", final)
            if cv.waitKey(1) == 27:
                break
        cv.destroyAllWindows()
        cv.namedWindow('Fusion', cv.WINDOW_NORMAL)
        cv.resizeWindow('Fusion', n, m)
        Rz_ = Rz
        Rx_ = Rx
        Ry_ = Ry
        Tz_ = Tz
        Ty_ = Ty
        Tx_ = Tx
        CutY_ = CutY
        CutZ_ = CutZ
        cv.createTrackbar('Rotation Z', 'Fusion', 250, 500, nothing)
        cv.createTrackbar('Rotation X', 'Fusion', 250, 500, nothing)
        cv.createTrackbar('Rotation Y', 'Fusion', 250, 500, nothing)
        cv.createTrackbar('Translation Z', 'Fusion', 20, 40, nothing)
        cv.createTrackbar('Translation Y', 'Fusion', 20, 40, nothing)
        cv.createTrackbar('Translation X', 'Fusion', 250, 500, nothing)
        cv.createTrackbar('Cut Z', 'Fusion', 250, 500, nothing)
        cv.createTrackbar('Cut Y', 'Fusion', 250, 500, nothing)
        while True:
            # Updating the parameters based on the trackbar positions
            Rz = (cv.getTrackbarPos('Rotation Z', 'Fusion') - 250) / 10 ** 7
            Rx = (cv.getTrackbarPos('Rotation X', 'Fusion') - 250) / 10 ** 7
            Ry = (cv.getTrackbarPos('Rotation Y', 'Fusion') - 250) / 10 ** 5
            Tz = cv.getTrackbarPos('Translation Z', 'Fusion') - 20
            Ty = cv.getTrackbarPos('Translation Y', 'Fusion') - 20
            Tx = (cv.getTrackbarPos('Translation X', 'Fusion')-250) / 1000
            CutY = cv.getTrackbarPos('Cut Y', 'Fusion') - 250
            CutZ = cv.getTrackbarPos('Cut Z', 'Fusion') - 250
            warp_matrix_rotation, _ = cv.Rodrigues(np.array([Rx+Rx_, Rz+Rz_, Ry+Ry_]))
            rotation = [Rx+Rx_, Rz+Rz_, Ry+Ry_]
            # warp_matrix_translation = np.array([[Tx+Tx_, 0., Ty+Ty_ / 1.], [0., Tx+Tx_, Tz+Tz_ / 1.]])
            warp_matrix_rotation[0, 0] = Tx+Tx_
            warp_matrix_rotation[1, 1] = Tx+Tx_
            warp_matrix_rotation[0, 2] = Ty+Ty_
            warp_matrix_rotation[1, 2] = Tz+Tz_

            imL_aligned = cv.warpPerspective(imL.copy(), warp_matrix_rotation, (n, m),
                                             flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
            imR_aligned = imR.copy()
            CutZ += CutZ_
            CutY += CutY_
            if CutZ >= 0:
                if CutY >= 0:
                    imR_aligned, imL_aligned = imR_aligned[CutZ:, CutY:], imL_aligned[CutZ:, CutY:]

                else:
                    imR_aligned, imL_aligned = imR_aligned[CutZ:, :CutY], imL_aligned[CutZ:, :CutY]
            else:
                if CutY >= 0:
                    imR_aligned, imL_aligned = imR_aligned[:CutZ, CutY:], imL_aligned[:CutZ, CutY:]
                else:
                    imR_aligned, imL_aligned = imR_aligned[:CutZ, :CutY], imL_aligned[:CutZ, :CutY]
            sample = {
                'imgL': imL_aligned / 255,
                'imgR': imR_aligned / 255
            }
            image_outputs = test_sample(sample, model)
            disparity = image_outputs["disp_est"]
            if m > 480:
                disparity = disparity / 140
            else:
                disparity = disparity / 70
            final = np.uint8((imR_aligned / 2 + imL_aligned / 2))
            final = cv.resize(final, (disparity.shape[1], disparity.shape[0]))
            final = np.hstack(
                [cv.cvtColor(final, cv.COLOR_RGB2BGR) / 255, np.stack([disparity, disparity, disparity], axis=2)])
            cv.imshow("Fusion", final)
            if cv.waitKey(1) == 27:
                break
        if len(imL.shape) > 2:
            cv.imwrite(path_save + '/left_rect.png', cv.cvtColor(imL_aligned, cv.COLOR_RGB2BGR))
        else:
            cv.imwrite(path_save + '/left_rect.png', imL_aligned)
        if len(imR.shape) > 2:
            cv.imwrite(path_save + '/right_rect.png', cv.cvtColor(imR_aligned, cv.COLOR_RGB2BGR))
        else:
            cv.imwrite(path_save + '/right_rect.png', cv.cvtColor(imR_aligned, cv.COLOR_RGB2BGR))
        with open(join(path_save, "transform_matrix"), "wb") as p:
            pickle.dump({"homography": warp_matrix_rotation,
                         "rotation": rotation,
                         "CutY": CutY,
                         "CutZ": CutZ}, p)
    cv.destroyAllWindows()
    return warp_matrix_rotation, CutY, CutZ


def SIFT(image_dst, image_src, MIN_MATCH_COUNT=4, matcher='FLANN', name='00', lowe_ratio=0,
         nfeatures=0, nOctaveLayers=4, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6, toggle_evaluate=False, verbose=False,
         delta=None):
    if delta is None:
        delta_x_min = 0
        delta_x_max = 1000
        delta_y_min = 0
        delta_y_max = 1000
    else:
        delta_x_min = delta[0]
        delta_x_max = delta[1]
        delta_y_min = delta[2]
        delta_y_max = delta[3]
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
        ratio_matches = 0.4
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
                src_temp = np.float32(kp_src[m.trainIdx].pt).reshape(-1, 2)
                dst_temp = np.float32(kp_dst[m.queryIdx].pt).reshape(-1, 2)
                if delta_y_max >= abs(src_temp[0][1] - dst_temp[0][1]) > delta_y_min and \
                        delta_x_max >= abs(src_temp[0][0] - dst_temp[0][0]) > delta_x_min:
                    good.append(m)
    if len(good) >= 0:
        src_pts = np.float32([kp_src[m.trainIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp_dst[m.queryIdx].pt for m in good]).reshape(-1, 2)
        #         M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        #         matchesMask = mask.ravel().tolist()
        pts_src = np.int32(src_pts)
        pts_dst = np.int32(dst_pts)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
    if toggle_evaluate:
        return len(kp_src), len(kp_dst), len(src_pts), t1, t2
    elif verbose:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)
        img3 = cv.drawMatches(image_dst, kp_dst, image_src, kp_src, good, None, **draw_params)
        # plt.figure(figsize=(30, 30), dpi=40)
        # plt.imshow(img3, 'gray'), plt.show()
        cv.putText(img3, f'Threshold min : {delta_x_min}, threshold max :{delta_x_max}', (10, 920), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 0, 0), 0)
        if img3.shape[0] > 500:
            img3 = cv.pyrDown(cv.cvtColor(img3, cv.COLOR_RGB2BGR))
        else:
            img3 = cv.cvtColor(img3, cv.COLOR_RGB2BGR)
        cv.imshow('matched points', img3)
        cv.waitKey(0)
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


def automatic_crop(image1, image2, origin='tl', mode="crop"):
    """
    :param image1: image1 input (numpy object or ImageCustom object)
    :param image2: image2 input (numpy object or ImageCustom object)
    :param origin: point of origin (top left (default): tl, bottom left: bl, top right: tr, bottom right: br)
    :param mode: "crop" for cropping output images to the smaller size, "full" to output the full images with zeros padding to match the size
    :return: cropped image1 and image2 according the maximum dimension from the origin point
    """
    h1, w1 = image1.shape[0], image1.shape[1]
    h2, w2 = image2.shape[0], image2.shape[1]
    if mode != "crop":
        if len(image1.shape)>2:
            out1 = np.zeros([max(h1, h2), max(w1, w2), 3])
        else:
            out1 = np.zeros([max(h1, h2), max(w1, w2)])
        if len(image2.shape)>2:
            out2 = np.zeros([max(h1, h2), max(w1, w2), 3])
        else:
            out2 = np.zeros([max(h1, h2), max(w1, w2)])
    if origin == 'tl':
        if mode=="crop":
            out1 = image1[:min(h1, h2), :min(w1, w2)]
            out2 = image2[:min(h1, h2), :min(w1, w2)]
        else:
            out1[:h1, :w1] = image1
            out2[:h2, :w2] = image2
    elif origin =='bl':
        if mode=="crop":
            out1 = image1[-min(h1, h2):, :min(w1, w2)]
            out2 = image2[-min(h1, h2):, :min(w1, w2)]
        else:
            out1[-h1:, :w1] = image1
            out2[-h2:, :w2] = image2

    elif origin == 'tr':
        if mode=="crop":
            out1 = image1[:min(h1, h2), -min(w1, w2):]
            out2 = image2[:min(h1, h2), -min(w1, w2):]
        else:
            out1[h1:, -w1:] = image1
            out2[h2:, -w2:] = image2

    elif origin =='br':
        if mode=="crop":
            out1 = image1[-min(h1, h2):, -min(w1, w2):]
            out2 = image2[-min(h1, h2):, -min(w1, w2):]
        else:
            out1[-h1:, -w1:] = image1
            out2[-h2:, -w2:] = image2

    else:
        if mode == "crop":
            out1 = image1[:min(h1, h2), :min(w1, w2)]
            out2 = image2[:min(h1, h2), :min(w1, w2)]
        else:
            out1[:h1, :w1] = image1
            out2[:h2, :w2] = image2
    return ImageCustom(out1, image1), ImageCustom(out2, image2)
