import os
from os.path import join
import cv2 as cv
import numpy as np


def dataloader(source_path, source):
    if source == 'lynred' or source == 'lynred_inf':
        if source == 'lynred':
            source_path = join(source_path, 'visible')
        else:
            source_path = join(source_path, 'infrared')
        fold = np.random.randint(0, 1)
        if fold:
            p_drive = join(source_path, 'Day')
        else:
            p_drive = join(source_path, 'Night')
        p_drive_left = join(p_drive, 'left')
        p_drive_right = join(p_drive, 'right')
        if len(os.listdir(p_drive_left)) - 1:
            n = np.random.randint(0, len(os.listdir(p_drive_left)) - 1)
        else:
            n = 0
        l_path = join(p_drive_left, os.listdir(p_drive_left)[n])
        r_path = join(p_drive_right, os.listdir(p_drive_right)[n])
    elif source == 'teddy':
        l_path = join(source_path, 'left_teddy.png')
        r_path = join(source_path, 'right_teddy.png')
    elif source == 'cones':
        l_path = join(source_path, 'left_cones.png')
        r_path = join(source_path, 'right_cones.png')
    else:
        l_path = join(source_path, 'left_cones.png')
        r_path = join(source_path, 'right_cones.png')

    sample = {
        'imgL': cv.imread(l_path, 1)/255,
        'imgR': cv.imread(r_path, 1)/255
    }
    return sample
