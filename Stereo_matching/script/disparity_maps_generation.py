import pathlib
import sys
import os

import numpy as np

from Stereo_matching.NeuralNetwork.monodepth2.Monodepth2 import parse_args, test_simple

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
import argparse
from Stereo_matching.NeuralNetwork.ACVNet_main.ACVNet_test import ACVNet_test
"""
   #################################################################################
   SUMMARY:
   This Script is used to generate all the disparity maps from all the pre-aligned images.
   It generate a disparity maps for each couple master/slave in visible AND infrared.
   The so-generated disparity are compared between visible and infrared.
   If the option has been set the disparity images will be stored in the 
   "'Day or Night'/'visible or infrared'/disparity_maps" folders.
   """


def disparity_maps_generation(sample=None, Time=None, image_type=None, monocular=False, post_process=70,
                              clean=False, verbose=False, path_save=None):
    if Time is None:
        Time = ['Day', 'Night']
    if image_type is None:
        image_type = ['visible', 'infrared', 'hybrid']
    for T in Time:
        for im_type in image_type:
            if not monocular:
                if sample is None:
                    print(f"\n  Computation of the disparity map for {T} {im_type} images ...")
                    ACVNet_test(None, verbose, post_process=post_process, time_of_day=T, im_type=im_type, clean=clean, path_save=path_save)
                else:
                    print(f"\n  Computation of the disparity map for the {T} {im_type} images sample ...")
                    _, _, disparity, m, M = ACVNet_test(sample, verbose, post_process=post_process, time_of_day=T, im_type=im_type, clean=False, path_save=path_save)
                    return disparity * (M - m) + m
            else:
                if sample is None:
                    print(f"\n  Computation of the monocular disparity map for {T} {im_type} images ...")
                    test_simple("mono+stereo_1024x320", '/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/hybrid/right',
                output_path='/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/hybrid/monocular_disparity_maps',
                            ext='png', no_cuda=False, no_save=False, pred_metric_depth=False)
                else:
                    print(f"\n  Computation of the monocular disparity map for the {T} {im_type} images sample ...")
                    disparity = test_simple("mono+stereo_1024x320", sample, output_path=None, ext='png', no_cuda=False,
                                            no_save=True, pred_metric_depth=False)
                    return disparity


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    """
    These arguments set the method and the source for the disparity estimation
    """
    parser.add_argument('--time', default=None, help='Either Day or Night, usefull only if an index is set')
    parser.add_argument('--type', default=None, help='Either visible or infrared, usefull only if an index is set')
    parser.add_argument('--mode', default=0, type=int, help='0 or 1, the basic mode, or the hybrid mode')
    parser.add_argument('--monocular', action='store_true',
                        help='Use the monocular depth projection instead of the classical stereo')
    """
    These arguments set the parameters for the disparity maps completion or post processing
    """
    parser.add_argument('--post_process', default=70, type=int,
                        help='Post process threshold the disparity map and remove the '
                             'outlier')
    parser.add_argument('--verbose', action='store_true', help='Show or not the results along the different steps')
    parser.add_argument('--clean', action='store_true', help='Clean the directory before to save the disparity maps')
    parser.add_argument('--path_video', default=None, help='Record all the disparity maps at the specified path')

    args = parser.parse_args()
    mode = args.mode
    post_process = args.post_process
    Time = args.time
    image_type = args.type
    translation_ratio = args.ratio
    verbose = args.verbose
    clean = args.clean
    monocular = args.monocular
    path_save = args.path_video

    disparity_maps_generation(sample=None, Time=Time, image_type=image_type,
                              monocular=monocular, post_process=post_process, clean=clean, verbose=verbose, path_save=path_save)
