import sys
import os
import argparse
import time

import cv2 as cv
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
from FUSION.script.image_management import name_generator, images_registration
from FUSION.classes.Image import ImageCustom
from FUSION.tools.registration_tools import *
from Stereo_matching.script.paralaxe_correction import paralax_correction
from Stereo_matching import __method_stereo__, __source_stereo__, __path_folder__
from Stereo_matching import *
from Stereo_matching.Tools.disparity_tools import disparity_post_process, reprojection_disparity

"""
   #################################################################################
   SUMMARY:
   Mode 1 :
   This script load all the images from the visible and infrared directories, compute all the disparity maps and 
   save them in the chronological order in the folder "video-frame"
   Mode 2 :
   This script load all the images from the infrared directories, compute all the projected disparity maps from the visible stereo pair 
   and save the infrared-distance fusion map in the folder "demo_projection"
   #############################################
   
   """
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    """
    These arguments set the method and the source for the disparity estimation
    """
    parser.add_argument('--mode', default=0, type=int, help='0 : Full images homography + Disparity + Video encoding'
                                                            '1 : Compute the projected disparity maps from the visible master onto the infrared master'
                                                            '2+ : Encode the comparison video between monocular depth estimation, and ACVNet')
    parser.add_argument('--step', default=0, type=int, help='0 means all steps, 1, 2, 3 are the other possibilities')
    parser.add_argument('--copy_disparity', action='store_true', help='Doesnt recompute the hybrid disparity, copy them from the other folders')
    """
    These arguments set the different parameter of the disparity estimation
    Some arguments are called only by a specific method, it wont be used if the method called is not the good one
    """
    parser.add_argument('--ratio', default=127 / (214 + 127), type=float,
                        help='Translation ratio (1 means full translation, - 1 inverse translation)')
    """
    These arguments set the parameters for the disparity maps completion or post processing
    """
    parser.add_argument('--post_process', default=70, type=int,
                        help='Post process threshold the disparity map and remove the '
                             'outlier')
    parser.add_argument('--clean', action='store_true', help='Clean the directory before to save the disparity maps')

    args = parser.parse_args()
    mode = args.mode
    post_process = args.post_process
    translation_ratio = args.ratio
    clean = args.clean
    step = args.step
    copy_disparity = args.copy_disparity
    Time = ['Night', 'Day']
    image_type = ['visible']#, 'visible', 'hybrid']
    path_save = "/home/godeta/PycharmProjects/LYNRED/Video_frame"

    if mode == 0:
        from Stereo_matching.NeuralNetwork.ACVNet_main.ACVNet_test import ACVNet_test
        if step == 0 or step == 1:
            print(f"\n1) Registration of the frames...")
            images_registration(verbose=False, Calibrate=False, Clean=True, path_s=path_save, new_random=False)
        else:
            print("Ignoring step 1")

        if step == 0 or step == 2:
            print(f"\n2) Computation of the disparity maps...")
            for T in Time:
                for im_type in image_type:
                    k = 1
                    print(f"\n2.{k}) {T}-{im_type} are processed...")
                    k += 1
                    ACVNet_test(None, False, post_process=post_process, time_of_day=T, im_type=im_type,
                                clean=True, path_save=path_save, colormap='viridis')
        else:
            print("Ignoring step 2")

        if step == 0 or step == 3:
            print(f"\n3) Video realisation...")
            for t in Time:
                video_array = []
                print(f"\n  {t} frames concatenation...")
                for idx, filename in tqdm(enumerate(sorted(os.listdir(join(path_save, t, "visible", "disparity_maps"))))):
                    map_inf = cv.imread(join(path_save, t, "infrared", "disparity_maps", filename), 1)
                    cv.putText(map_inf, 'Disparity from infrared stereo, ' + t, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                               (255, 255, 255), 1)
                    map_vis = cv.imread(join(path_save, t, "visible", "disparity_maps", filename), 1)
                    cv.putText(map_vis, 'Disparity from visible stereo, ' + t, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                               (255, 255, 255), 1)
                    img_inf = cv.imread(join(path_save, t, "infrared", "left", "left" + name_generator(idx) + '.png'), 0)
                    img_inf = np.stack([img_inf, img_inf, img_inf], axis=2)
                    cv.putText(img_inf, 'Left infrared source, ' + t, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                               (255, 255, 255), 1)
                    img_vis = cv.imread(join(path_save, t, "visible", "left", "left" + name_generator(idx) + '.png'), 1)
                    cv.putText(img_vis, 'Left visible source, ' + t, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                               (255, 255, 255), 1)
                    inf = np.vstack([img_inf, map_inf])
                    vis = np.vstack([img_vis, map_vis])
                    frame = np.hstack([vis, inf])
                    height, width, _ = frame.shape
                    size = (width, height)
                    video_array.append(frame)
                fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
                out = cv.VideoWriter('/home/godeta/PycharmProjects/LYNRED/Video_frame/Disparity_' + t + '.mp4', fourcc, 30, size)

                print(f"\n  Video encoding...")
                for frame in tqdm(video_array):
                    out.write(frame)
                out.release()
        else:
            print("Ignoring step 3")
    elif mode == 1:
        paralax_correction(post_process=post_process, index=-1, Time=Time, translation_ratio=translation_ratio, verbose=False,
                           clean=True, step=step, new_random=False, calibrate=False, monocular=False, folder=path_save,
                           path_video=path_save, mode='video')
        print(f"Step 4) Frame concatenations :")
        for t in Time:
            video_array = []
            print(f"\n  {t} frames concatenation...")
            for idx, filename in tqdm(enumerate(sorted(os.listdir(join(path_save, t, "hybrid", "infrared_projected"))))):
                img_inf_projected = ImageCustom(join(path_save, t, "hybrid", "infrared_projected", filename))
                img_inf = ImageCustom(join(path_save, t, "hybrid", "left", "left" + name_generator(idx) + '.png'))
                img_vis = ImageCustom(join(path_save, t, "hybrid", "right", "right" + name_generator(idx) + '.png'))
                fus = (img_vis * 0.5 + img_inf * 0.5) / 255
                fus = fus.BGR()
                fus_corrected = (img_vis * 0.5 + img_inf_projected * 0.5) / 255
                fus_corrected = fus_corrected.BGR()
                cv.putText(fus, 'Image fusion without parallax rectification, ' + t, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 1)
                cv.putText(fus_corrected, 'Image fusion with parallax rectification, ' + t, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 1)
                frame = np.hstack([fus, fus_corrected])
                height, width, _ = frame.shape
                size = (width, height)
                video_array.append(frame)
            fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
            out = cv.VideoWriter('/home/godeta/PycharmProjects/LYNRED/Video_frame/Parallax Correction ' + t + '.mp4',
                                 fourcc, 30, size)

            print(f"\n  Video encoding...")
            for frame in tqdm(video_array):
                out.write(frame)
            out.release()
    else:
        for t in Time:
            video_array = []
            print(f"\n  {t} frames concatenation...")
            for idx, filename in tqdm(enumerate(sorted(os.listdir(join(path_save, t, "visible", "monocular_disparity"))))):
                map_vis = cv.imread(join(path_save, t, "visible", "monocular_disparity", filename), 1)
                cv.putText(map_vis, 'Disparity from visible left image only, ' + t, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 1)
                img_vis = cv.imread(join(path_save, t, "visible", "disparity_maps", "disparity" + name_generator(idx) + '.png'), 1)
                cv.putText(img_vis, 'Left visible source, ' + t, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 1)
                frame = np.hstack([img_vis, map_vis])
                height, width, _ = frame.shape
                size = (width, height)
                video_array.append(frame)
            fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
            out = cv.VideoWriter('/home/godeta/PycharmProjects/LYNRED/Video_frame/Disparity_method_comparison_' + t + '.mp4', fourcc, 30,
                                 size)

            print(f"\n  Video encoding...")
            for frame in tqdm(video_array):
                out.write(frame)
            out.release()
