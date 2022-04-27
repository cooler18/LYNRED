import argparse
import pickle
import time
from os.path import join

import cv2 as cv
import timeit
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import area_closing, closing, square, cube, disk, ball
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, normalized_mutual_information


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


def automatic_registration(imL, imR, matrix, nameL ="left_rect.png", nameR="right_rect.png"):
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
    p = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/visible/Day"
    cv.imwrite(p + '/left/' + nameL, cv.cvtColor(imL_aligned, cv.COLOR_RGB2BGR))
    cv.imwrite(p + '/right/' + nameR, cv.cvtColor(imR_aligned, cv.COLOR_RGB2BGR))


def manual_registration(imL, imR):
    def nothing(x):
        pass
    cv.namedWindow('Fusion', cv.WINDOW_NORMAL)
    m, n = imL.shape[:2]
    cv.resizeWindow('Fusion', n, m)
    cv.createTrackbar('Rotation Z', 'Fusion', 250, 500, nothing)
    cv.createTrackbar('Rotation X', 'Fusion', 250, 500, nothing)
    cv.createTrackbar('Rotation Y', 'Fusion', 250, 500, nothing)
    cv.createTrackbar('Translation Z', 'Fusion', 250, 500, nothing)
    cv.createTrackbar('Translation Y', 'Fusion', 250, 500, nothing)
    cv.createTrackbar('Translation X', 'Fusion', 250, 500, nothing)
    cv.createTrackbar('Cut Z', 'Fusion', 250, 500, nothing)
    cv.createTrackbar('Cut Y', 'Fusion', 250, 500, nothing)
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
        final = np.uint8((imR_aligned / 2 + imL_aligned / 2))
        cv.imshow("Fusion", cv.cvtColor(final, cv.COLOR_RGB2BGR)/255)
        if cv.waitKey(1) == 27:
            break
    p = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/visible/Day"
    cv.imwrite(p + '/left/left_rect.png', cv.cvtColor(imL_aligned, cv.COLOR_RGB2BGR))
    cv.imwrite(p + '/right/right_rect.png', cv.cvtColor(imR_aligned, cv.COLOR_RGB2BGR))
    with open(join(p, "Calibration", "transform_matrix_slaveToMaster_vis"), "wb") as p:
        pickle.dump({"matrix_rotation": warp_matrix_rotation,
                     "matrix_translation": warp_matrix_translation,
                     "CutY": CutY,
                     "CutZ": CutZ}, p)
    cv.destroyAllWindows()
    return warp_matrix_rotation, warp_matrix_translation, CutY, CutZ


def image_registration(im1, im2, warp_mode):

    # Convert images to grayscale
    im1_gray = im1.GRAYSCALE()
    im2_gray = im2.GRAYSCALE()

    # Find size of image1
    sz = im1.shape

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 100

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 3)
    print(warp_matrix)
    if warp_mode == cv.MOTION_HOMOGRAPHY:
    # Use warpPerspective for Homography
        im2_aligned = cv.warpPerspective(im2, warp_matrix, (sz[1], sz[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
    else:
    # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

    # Show final results
    cv.imshow("Image 1", im1.BGR())
    cv.imshow("Image 2", im2.BGR())
    cv.imshow("Aligned Image 2", (im2_aligned/2 + im1/2)/255)
    cv.waitKey(0)
    cv.imwrite('/home/godeta/PycharmProjects/LYNRED/LynredDataset/visible/Day/left/left_rect.png', im1.BGR())
    cv.imwrite('/home/godeta/PycharmProjects/LYNRED/LynredDataset/visible/Day/right/right_rect.png', cv.cvtColor(im2_aligned, cv.COLOR_RGB2BGR))
    cv.destroyAllWindows()
