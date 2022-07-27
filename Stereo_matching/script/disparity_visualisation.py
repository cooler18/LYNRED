import os
import pathlib
import pickle
import argparse
import sys
from os.path import join

import cv2 as cv
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
from FUSION.script.image_management import name_generator
from Stereo_matching import reconstruction_from_disparity


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--time', default='Night', help='Either Day or Night, usefull only if an index is set')
    parser.add_argument('--index', default=-1, type=int, help='idx of the wanted picture (0-99) // -1 means random')
    parser.add_argument('--mode', default=0, type=int, help=
    'mode 0 : visualize the disparity for both master visible and infrared, '
    'mode 1 : visualize the disparity map of the master infrared to the master visible')
    args = parser.parse_args()
    index = args.index
    Time = args.time
    mode = args.mode
    ############################################################
    if index == -1 or index > 99 or index < 0:
        number = np.random.randint(0, 99)
    else:
        number = index
    number = k_str = name_generator(number)
    base_path = os.path.dirname(os.path.dirname(pathlib.Path().resolve()))
    ############################################################
    if mode == 0:
        print(f"Image n°{number}")
        with open(
                "/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + Time + "/visible/disparity_maps/disparity" + number,
                "rb") as f:
            disparity = pickle.load(f)
        with open(
                "/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + Time + "/infrared/disparity_maps/disparity" + number,
                "rb") as f:
            disparity_inf = pickle.load(f)
        with open(
                "/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + Time + "/hybrid/disparity_maps/disparity" + number,
                "rb") as f:
            disparity_hyb = pickle.load(f)
        disp_vis = np.array(disparity)
        disp_inf = np.array(disparity_inf)
        disparity_hyb = np.array(disparity_hyb)
        vis = cv.imread(
            "/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + Time + "/visible/right/right" + number + '.png', 1)
        inf = cv.imread(
            "/home/godeta/PycharmProjects/LYNRED/LynredDataset/" + Time + "/infrared/right/right" + number + '.png', 0)

        cv.imshow('visible disparity', disp_vis / 70)
        cv.imshow('infrared disparity', disp_inf / 70)
        cv.imshow('hybrid disparity', disparity_hyb / 70)
        cv.imshow('visible source', vis)
        cv.imshow('infrared source', inf)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        with open(join(base_path, "LynredDataset", Time + "/hybrid/disparity_maps/disparity" + number), "rb") as f:
            disparity = np.array(pickle.load(f))
        m, M = disparity.min(), disparity.max()
        disparity = (disparity - m)/(M - m)

        vis = cv.imread(join(base_path, "LynredDataset", Time + "/hybrid/right/right" + number + '.png'), 1)
        inf = cv.imread(join(base_path, "LynredDataset", Time + "/hybrid/left/left" + number + '.png'), 0)
        print(disparity.shape, inf.shape)
        # image_corrected = reconstruction_from_disparity(inf, inf, disparity, m, M, 0,
        #                                                 False, True, False, True, False, orientation=0)

        cv.imshow('infrared disparity', disparity / 66)
        cv.imshow('visible source', vis)
        cv.imshow('infrared source', inf)
        # cv.imshow('Infrared image corrected', image_corrected)
        cv.waitKey(0)
        cv.destroyAllWindows()
