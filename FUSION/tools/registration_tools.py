import os
import pathlib
import pickle
from os.path import join
import cv2 as cv
import timeit
import matplotlib.pyplot as plt
import numpy as np


def automatic_registration(imL, imR, matrix, path_save, nameL ="left_rect.png", nameR="right_rect.png", save=True):
    warp_matrix_rotation, warp_matrix_translation = matrix["matrix_rotation"], matrix["matrix_translation"]
    CutY, CutZ = matrix["CutY"], matrix["CutZ"]
    m, n = imL.shape[:2]
    imL_aligned = cv.warpPerspective(imL.copy(), warp_matrix_rotation, (n, m),
                                     flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
    imL_aligned = cv.warpAffine(imL_aligned, warp_matrix_translation, (n, m),
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
    if save:
        if os.path.basename(os.path.dirname(os.path.dirname(path_save))) == "Video_frame":
            if imL_aligned.shape[1] > 640:
                imL_aligned = cv.pyrDown(imL_aligned)
            if imR_aligned.shape[1] > 640:
                imR_aligned = cv.pyrDown(imR_aligned)
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


def manual_registration(imL, imR, path_save):
    from Stereo_matching.NeuralNetwork.ACVNet_main.ACVNet_main import test_sample
    from Stereo_matching.NeuralNetwork.ACVNet_main.ACVNet_test import initialize_model
    model = initialize_model(verbose=False)

    def nothing(x):
        pass
    try:
        with open(join(path_save, "transform_matrix"), "rb") as p:
            matrix = pickle.load(p)
        warp_matrix_rotation = matrix["matrix_rotation"]
        warp_matrix_rotation = cv.Rodrigues(warp_matrix_rotation)
        Rx, Rz, Ry = warp_matrix_rotation[0]
        warp_matrix_translation = matrix["matrix_translation"]
        Tx, Ty, Tz = int((warp_matrix_translation[0, 0] - 0.5) * 500), warp_matrix_translation[0, 2], warp_matrix_translation[1, 2]
        CutY = matrix["CutY"]
        CutZ = matrix["CutZ"]
        with open(join(path_save, "transform_matrix_old"), "wb") as p:
            pickle.dump(matrix, p)
    except FileNotFoundError:
        print('No disparity matrix found')
        Rx, Rz, Ry = 0, 0, 0
        Tx, Ty, Tz = 250, 0, 0
        CutY = 0
        CutZ = 0
    m, n = imL.shape[:2]
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
        warp_matrix_translation = np.array([[Tx, 0., Ty/1.], [0., Tx, Tz/1.]])
        imL_aligned = cv.warpPerspective(imL.copy(), warp_matrix_rotation, (n, m), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
        imL_aligned = cv.warpAffine(imL_aligned, warp_matrix_translation, (n, m), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
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
        warp_matrix_translation = np.array([[Tx+Tx_, 0., Ty+Ty_ / 1.], [0., Tx+Tx_, Tz+Tz_ / 1.]])
        imL_aligned = cv.warpPerspective(imL.copy(), warp_matrix_rotation, (n, m),
                                         flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
        imL_aligned = cv.warpAffine(imL_aligned, warp_matrix_translation, (n, m),
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
        pickle.dump({"matrix_rotation": warp_matrix_rotation,
                     "matrix_translation": warp_matrix_translation,
                     "CutY": CutY,
                     "CutZ": CutZ}, p)
    cv.destroyAllWindows()
    return warp_matrix_rotation, warp_matrix_translation, CutY, CutZ


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