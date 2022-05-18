import pickle
import shutil
from os.path import join

import argparse
import cv2 as cv
import pathlib
import os
import sys

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
from FUSION.classes.Image import *


# from .. classes.Image import ImageCustom

def name_generator(idx):
    if idx < 10000:
        k_str = '0' + str(idx)
        if idx < 1000:
            k_str = '0' + k_str
            if idx < 100:
                k_str = '0' + k_str
                if idx < 10:
                    k_str = '0' + k_str
    else:
        k_str = str(idx)
    return k_str


# from FUSION.tools.data_management_tools import *

# from matplotlib.colors import ListedColormap
# import matplotlib.cm
#####################################################################
# Script to copy all infrared/visible images from the multiple folder of "Images" to a single one "Image grouped"
# DIR = "D:\Travail\LYNRED\Images"
# DEST_DIR = "D:\Travail\LYNRED\FUSION\Images_grouped"
#
# copy_image(DIR, DEST_DIR, init=True)
####################################################################
# # Script to generate the bar plot of the colorspace
# with open('D:\Travail\LYNRED\FUSION\interface\data_interface\LUT8_lifeinred_color.dat') as f:
#     lines = f.readlines()
# name = lines[0].split()[1]
# cmap = np.zeros([256, 4])
# for l in lines[4:]:
#     idx, r, g, b, a = int(l.split()[0]), float(l.split()[1])/255, float(l.split()[2])/255, float(l.split()[3])/255, 1
#     cmap[idx] = r, g, b, a
# Lynred_cmap = ListedColormap(cmap, name=name)
# matplotlib.cm.register_cmap(name=name, cmap=Lynred_cmap)
# plot_color_gradients('Perceptually Uniform Sequential',
#                      ['lifeinred_color'])#, 'plasma', 'inferno', 'magma', 'cividis'])

#######################################################################
# Script to generate the train and label for RGB coloration of IR image by CNN
# SRC_DIR = "D:\Travail\LYNRED\FUSION\Images_grouped"
# concatane_train_label(SRC_DIR, n=20)

#########################################################################

# Code for automatic images registration
def images_registration(Time=None, image_type=None, verbose=False, Calibrate = False, Clean = False, path_s=None, new_random=False, save=True, index=-1):
    from FUSION.tools.registration_tools import manual_registration, automatic_registration
    ## SCRIPT GLOBAL OPTIONS #####
    if Time is None:
        Time = ["Day", "Night"]
    else:
        if not Calibrate and not isinstance(Time, list):
            Time = [Time]

    if image_type is None:
        image_type = ["visible", "infrared", "hybrid"]
    else:
        if not Calibrate:
            image_type = [image_type]
    base_path = os.path.dirname(os.path.dirname(pathlib.Path().resolve()))
    number_of_images = 100
    if index == -1:
        if new_random:
            n = np.random.randint(0, len(os.listdir(join(base_path, 'Images', 'Day', 'master', 'visible'))), number_of_images)
            with open(join(base_path, "LynredDataset", "list_number"), "wb") as f:
                pickle.dump(n, f)
        elif path_s:
            n = range(0, 3600, 1)
            number_of_images = 3600
            base_path = os.path.dirname(path_s)
        else:
            with open(join(base_path, "LynredDataset", "list_number"), "rb") as f:
                n = pickle.load(f)
    else:
        n = [index]
        number_of_images = 1
    if not Calibrate:
        for t in Time:
            for im_type in image_type:
                print(f"{number_of_images} of {t} - {im_type} are being processed...")
                p = join(base_path, "Images", t)
                if path_s is None:
                    path_save = join(base_path, "LynredDataset", t, im_type)
                else:
                    path_save = join(path_s, t, im_type)
                if Clean:
                    for filename in os.listdir(join(path_save, 'left')):
                        file_path = os.path.join(path_save, 'left', filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            # elif os.path.isdir(file_path):
                            #     shutil.rmtree(file_path)
                        except Exception as e:
                            print('Failed to delete %s. Reason: %s' % (file_path, e))
                    for filename in os.listdir(join(path_save, 'right')):
                        file_path = os.path.join(path_save, 'right', filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            print('Failed to delete %s. Reason: %s' % (file_path, e))
                ## Transfert Matrix Loading
                with open(join(base_path, "LynredDataset", t, im_type, "Calibration/transform_matrix"), "rb") as f:
                    matrix = pickle.load(f)

                if im_type == 'infrared':
                    im_type = 'infrared_corrected'
                if im_type != "hybrid":
                    list_left = sorted(os.listdir(join(p, 'master', im_type)))
                    list_right = sorted(os.listdir(join(p, 'slave', im_type)))
                else:
                    list_left = sorted(os.listdir(join(p, 'master', "infrared_corrected")))
                    list_right = sorted(os.listdir(join(p, 'slave', "visible")))

                for i in tqdm(range(number_of_images)):
                    if im_type != "hybrid":
                        imgR = join(p, 'slave', im_type, list_right[n[i]])
                        imgL = join(p, 'master', im_type, list_left[n[i]])
                        imgL = ImageCustom(imgL)
                        imgR = ImageCustom(imgR)
                    else:
                        imgR = join(p, 'slave', 'visible', list_right[n[i]])
                        imgL = join(p, 'master', 'infrared_corrected', list_left[n[i]])
                        imgL = ImageCustom(imgL)
                        imgR = cv.pyrDown(ImageCustom(imgR))
                    if verbose:
                        cv.imshow('control Left', ImageCustom(imgL).BGR())
                        cv.imshow('control Right', ImageCustom(imgR).BGR())
                        cv.waitKey(0)
                        cv.destroyAllWindows()
                    if save:
                        k = len(os.listdir(join(path_save, 'left')))
                        k_str = name_generator(k)
                        nameL = 'left' + k_str + '.png'
                        nameR = 'right' + k_str + '.png'
                        automatic_registration(imgL, imgR, matrix, path_save, nameL=nameL, nameR=nameR, save=save)
                    else:
                        imL_aligned, imR_aligned = automatic_registration(imgL, imgR, matrix, None, nameL=None, nameR=None, save=save)
                        return imL_aligned, imR_aligned
                print(f"Automatic registration of the {i + 1} images is done !")
    else:
        if image_type == "visible":
            if Time == "Day":
                imgL = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/visible/Calibration/Calib_left.png"
                imgR = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/visible/Calibration/Calib_right.png"
                path_save = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/visible/Calibration"
            else:
                imgL = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/visible/Calibration/Calib_left.png"
                imgR = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/visible/Calibration/Calib_right.png"
                path_save = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/visible/Calibration"
        elif image_type == 'infrared':
            if Time == "Day":
                imgL = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/infrared/Calibration/Calib_left.png"
                imgR = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/infrared/Calibration/Calib_right.png"
                path_save = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/infrared/Calibration"
            else:
                imgL = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/infrared/Calibration/Calib_left.png"
                imgR = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/infrared/Calibration/Calib_right.png"
                path_save = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/infrared/Calibration"
        else:
            if Time == "Day":
                imgL = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/hybrid/Calibration/Calib_left.png"
                imgR = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/hybrid/Calibration/Calib_right.png"
                path_save = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/hybrid/Calibration"
            else:
                imgL = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/hybrid/Calibration/Calib_left.png"
                imgR = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/hybrid/Calibration/Calib_right.png"
                path_save = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/hybrid/Calibration"
        if image_type == 'hybrid':
            imgL = ImageCustom(imgL).RGB()
            imgR = cv.pyrDown(ImageCustom(imgR))
        else:
            imgR = ImageCustom(imgR)
            imgL = ImageCustom(imgL)
        manual_registration(imgL, imgR, path_save)

#######################################################################
# Script to process the infrared images using the Lynred package
# The processed images will be stored in the same parent folder inside infrared_corrected

#########################################################################
#
# import subprocess
#
# # Path to a Python interpreter that runs any Python script
# # under the virtualenv /path/to/virtualenv/
# python_bin = "/home/godeta/PycharmProjects/LYNRED/venv/bin/python"
#
# # Path to the script that must run under the virtualenv
# script_file = "/home/godeta/PycharmProjects/LYNRED/FUSION/script/infrared_image_management.py"
#
# subprocess.Popen([python_bin, script_file])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    """
    These arguments set the method and the source for the image management scripts
    """
    parser.add_argument('--time', default=None, help='Either Day or Night')
    parser.add_argument('--type', default=None, help='Either visible or infrared')
    parser.add_argument('--calib', action='store_true', help='Calibrate the chosen images')
    """
    These arguments set the different parameter of the disparity estimation
    Some arguments are called only by a specific method, it wont be used if the method called is not the good one
    """

    """
    These arguments set the parameters for the disparity maps completion or post processing
    """
    parser.add_argument('--verbose', action='store_true', help='Show or not the results along the different steps')
    parser.add_argument('--clean', action='store_true', help='Clean the directory before to save the disparity maps')
    parser.add_argument('--new_rand', action='store_true', help='Generate a new serie of picture index')

    args = parser.parse_args()

    Time = args.time
    calibrate = args.calib
    im_type = args.type
    verbose = args.verbose
    clean = args.clean
    new_random = args.new_rand

    images_registration(Time=Time, image_type=im_type, verbose=verbose, Calibrate=calibrate, Clean=clean, path_s=None,
                        new_random=new_random)