import os
import pickle
import sys
from os.path import join
import cv2 as cv
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
from FUSION.classes.Image import ImageCustom
from Stereo_matching.Tools.disparity_tools import disparity_post_process


def dataloader(source_path, source, calib=False, Time='Day', n=-1):
    source_path = join(source_path, Time)
    if not calib:
        if source == 'lynred' or source == 'lynred_inf' or source == 'lynred_vis':
            if source == 'lynred_vis':
                p_drive = join(source_path, 'visible')
            elif source == 'lynred_inf':
                p_drive = join(source_path, 'infrared')
            else:
                p_drive = join(source_path, 'hybrid')
            p_drive_left = join(p_drive, 'left')
            p_drive_right = join(p_drive, 'right')

            if len(os.listdir(p_drive_left)) - 1 > 0 and n < 0:
                n = np.random.randint(0, len(os.listdir(p_drive_left)) - 1)
            else:
                n = n
            l_path = join(p_drive_left, sorted(os.listdir(p_drive_left))[n])
            r_path = join(p_drive_right, sorted(os.listdir(p_drive_right))[n])
            print(f"    Index of the source images : {n}")
        elif source == 'teddy':
            l_path = join("/home/godeta/PycharmProjects/LYNRED/Stereo_matching/Samples", 'left_teddy.png')
            r_path = join("/home/godeta/PycharmProjects/LYNRED/Stereo_matching/Samples", 'right_teddy.png')
        elif source == 'cones':
            l_path = join("/home/godeta/PycharmProjects/LYNRED/Stereo_matching/Samples", 'left_cones.png')
            r_path = join("/home/godeta/PycharmProjects/LYNRED/Stereo_matching/Samples", 'right_cones.png')
        else:
            l_path = join("/home/godeta/PycharmProjects/LYNRED/Stereo_matching/Samples", 'left_cones.png')
            r_path = join("/home/godeta/PycharmProjects/LYNRED/Stereo_matching/Samples", 'right_cones.png')
    else:
        if source == 'lynred_vis':
            if Time == 'Day':
                l_path = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/visible/Calibration/left_rect.png"
                r_path = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/visible/Calibration/right_rect.png"
            else:
                l_path = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/visible/Calibration/left_rect.png"
                r_path = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/visible/Calibration/right_rect.png"
        elif source == 'lynred_inf':
            if Time == 'Day':
                l_path = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/infrared/Calibration/left_rect.png"
                r_path = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/infrared/Calibration/right_rect.png"
            else:
                l_path = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/infrared/Calibration/left_rect.png"
                r_path = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/infrared/Calibration/right_rect.png"
        else:
            if Time == 'Day':
                l_path = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/hybrid/Calibration/left_rect.png"
                r_path = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/hybrid/Calibration/right_rect.png"
            else:
                l_path = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/hybrid/Calibration/left_rect.png"
                r_path = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/hybrid/Calibration/right_rect.png"
    if source == 'lynred_inf':
        sample = {
            'imgL': np.array(cv.imread(l_path, 0)) / 255,
            'imgR': np.array(cv.imread(r_path, 0)) / 255
        }
    elif source == "lynred":
        sample = {
            'imgL': np.array(cv.imread(l_path, 1)) / 255,
            'imgR': np.array(cv.imread(r_path, 0)) / 255
        }
    else:
        sample = {
            'imgL': cv.imread(l_path, 1)/255,
            'imgR': cv.imread(r_path, 1)/255
        }
    return sample


def data_superloader(time_of_day, im_type, post_process=0, clean=False, path_save=None, colormap='inferno'):
    if path_save is None:
        path_clean = join("/home/godeta/PycharmProjects/LYNRED/LynredDataset", time_of_day, im_type)
    else:
        path_clean = join(path_save, time_of_day, im_type)
    if clean:
        for filename in os.listdir(join(path_clean, "disparity_maps")):
            file_path = os.path.join(path_clean, "disparity_maps", filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def overload_func(func):
        def decorateur(*args, **kwargs):
            model = args[1]
            if im_type != 'hybrid':
                list_left = sorted(os.listdir(join(path_clean, 'left')))
                list_right = sorted(os.listdir(join(path_clean, 'right')))
            else:
                list_left = sorted(os.listdir(join(os.path.dirname(path_clean), 'visible', 'left')))
                list_right = sorted(os.listdir(join(os.path.dirname(path_clean), 'visible', 'right')))
            for idx, imL in tqdm(enumerate(list_left)):
                imR = list_right[idx]
                if im_type != 'hybrid':
                    path_L = join(path_clean, 'left', imL)
                    path_R = join(path_clean, 'right', imR)
                else:
                    path_L = join(os.path.dirname(path_clean), 'visible', 'left', imL)
                    path_R = join(os.path.dirname(path_clean), 'visible', 'right', imR)
                if im_type == 'infrared':
                    sample = {
                        'imgL': np.array(cv.imread(path_L, 0)) / 255,
                        'imgR': np.array(cv.imread(path_R, 0)) / 255
                    }
                else:
                    sample = {
                        'imgL': np.array(cv.imread(path_L, 1)) / 255,
                        'imgR': np.array(cv.imread(path_R, 1)) / 255
                    }
                disparity = func(sample, model)
                disparity = disparity_post_process(disparity, 0, 0, post_process)
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
                if path_save is None:
                    name = 'disparity' + k_str# + '.png'
                    with open(join(path_clean, "disparity_maps", name), "wb") as f:
                        pickle.dump(disparity, f)
                else:
                    name = 'disparity' + k_str + '.png'
                    disparity = ImageCustom(disparity/post_process*255).RGB(colormap=colormap).BGR()
                    cv.imwrite(join(path_clean, "disparity_maps", name), disparity)
        return decorateur
    return overload_func

