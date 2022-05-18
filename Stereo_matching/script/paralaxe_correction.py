import pathlib
import random
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
from Stereo_matching.script.disparity_maps_generation import disparity_maps_generation
import argparse
import cv2
from tqdm import tqdm
from Stereo_matching.Tools.disparity_tools import reprojection_disparity, reconstruction_from_disparity
from FUSION.script.image_management import name_generator, images_registration
from FUSION.tools.registration_tools import *
from FUSION.classes.Image import ImageCustom
from Stereo_matching import __path_folder__

"""
   #################################################################################
   SUMMARY:
   This Script present a processing pipe to handle the paralaxe correction
   Either for one picture referenced with an index, either for the the whole dataset.
   Requirements :
    - Path to a Dataset following this organisation :
   
    - The pictures have to be 640x480 or 1280x960 in entrance
    - Pictures from visible light (left and right for stereo method - only left for monocular method)
    - Picture from infrared light (only left)
    - The projection matrix from 
   """


def paralax_correction(post_process=70, index=-1, Time=None, translation_ratio=80 / (214 + 127), verbose=False,
                       clean=False,
                       step=0, new_random=False, calibrate=False, monocular=False, folder=None, path_video=None,
                       mode='image'):
    if Time is None:
        if index != -1:
            Time = ['Day']
        else:
            Time = ['Day', 'Night']
    elif isinstance(Time, list):
        pass
    else:
        Time = [Time]
    if index != -1:
        step = 0

    if mode == 'video' and path_video is None:
        raise ValueError('A path is needed for the video mode')

    #############################################################################################################
    # 1st Step : image registration
    # We need to register images to align Master visible onto Slave visible // master infrared onto salve visible
    # The Calibration will be set at the opening with the last parameters used
    # !! The calibration step will be automatically skip if the option --calib is not set !!
    # The registration will be compute on all the images in the --datapath
    # The result is stored in Time/hybrid/left:right
    #############################################################################################################
    if step == 0 or step == 1:
        print(f"\nStep 1) Images registration...")
        if calibrate:
            print(f"    Step 1.1) Calibration before registration")
            images_registration(Time=Time, image_type='hybrid', verbose=verbose, Calibrate=calibrate, Clean=False,
                                path_s=None, new_random=new_random)
            images_registration(Time=Time, image_type='visible', verbose=verbose, Calibrate=calibrate, Clean=False,
                                path_s=None, new_random=new_random)
        else:
            print(f"  Calibration has been skipped")
        if index == -1:
            images_registration(Time=Time, image_type='hybrid', verbose=False, Calibrate=False, Clean=True,
                                path_s=path_video, new_random=new_random)
            images_registration(Time=Time, image_type='visible', verbose=False, Calibrate=False, Clean=True,
                                path_s=path_video, new_random=new_random)
        else:
            im_infrared_aligned, im_visible_slave_aligned = \
                images_registration(Time=Time[0], image_type='hybrid', verbose=verbose, Calibrate=False, Clean=False,
                                    path_s=None, new_random=new_random, save=False, index=index)
            im_visible_master_aligned_aligned, im_visible_aligned = \
                images_registration(Time=Time[0], image_type='visible', verbose=verbose, Calibrate=False, Clean=False,
                                    path_s=None, new_random=new_random, save=False, index=index)
    else:
        print(f"\nStep 1) Images registration has been skipped")

    #############################################################################################################
    # 2nd Step : Disparity computation
    # We need to compute the disparity between the visible stereo images
    # The result will be the disparity of the master in compare to the slave visible camera
    # This result can be obtained with only one image by a monocular method
    # The result is stored in Time/hybrid/disparity_maps
    #############################################################################################################
    if step == 0 or step == 2:
        print(f"\nStep 2) Disparity computation from visible stereo...")
        if index == -1:
            disparity_maps_generation(sample=None, Time=Time, image_type=['hybrid'], monocular=monocular,
                                      post_process=post_process,
                                      clean=True, verbose=False, path_save=path_video)
        else:
            sample = {'imgL': im_visible_master_aligned_aligned / 255, 'imgR': im_visible_aligned / 255}
            disparity_visible_master = cv.pyrDown(
                disparity_maps_generation(sample=sample, Time=Time, image_type=['hybrid'], monocular=monocular,
                                          post_process=70, clean=False, verbose=verbose)) * 0.5
            if monocular:
                disparity_visible_master = disparity_visible_master * 0.7
    else:
        print("\nStep 2) Disparity computation has been skipped")
    #############################################################################################################
    # 3rd Step : Disparity projection & Final projection
    # We need to reproject the computed disparity maps onto the master IR image and then to project the IR image
    # onto the visible slave image
    # The result will be the fusion of infrared master images (left camera) with the slave visible image (right camera)
    #############################################################################################################
    if step == 0 or step == 3:
        print(f"\nStep 3) Disparity projection & Final projection...")
        if index != -1:
            print(folder)
            with open(join(folder, Time[0] + "/visible/Calibration/transform_matrix"), "rb") as f:
                matrix = pickle.load(f)
                cut_vis = (int(-matrix['CutZ'] / 2), int(-matrix['CutY'] / 2))
            with open(join(folder, Time[0] + "/hybrid/Calibration/transform_matrix"), "rb") as f:
                matrix = pickle.load(f)
                cut_inf = (-matrix['CutZ'], -matrix['CutY'])
            ################ Reprojection of the disparity maps and adjustment of the border #########################
            print(f"\n 3.1) Projection of the disparity map...")
            disparity_infrared_master = reprojection_disparity(disparity_visible_master, translation_ratio,
                                                               verbose=verbose)
            temp = np.zeros([480, 640])
            if cut_vis[0] >= 0:
                if cut_vis[1] >= 0:
                    temp[cut_vis[0]:, cut_vis[1]:] = disparity_infrared_master
                    temp[cut_vis[0]:, :cut_vis[1]] = disparity_infrared_master[:, :0] * np.ones(
                        [1, disparity_infrared_master.shape[0]])
                else:
                    temp[cut_vis[0]:, :cut_vis[1]] = disparity_infrared_master
                    temp[cut_vis[0]:, cut_vis[1]:] = disparity_infrared_master[:, -1:] * np.ones([1, -cut_vis[1]])
            else:
                if cut_vis[1] >= 0:
                    temp[:cut_vis[0], cut_vis[1]:] = disparity_infrared_master
                    temp[:cut_vis[0], :cut_vis[1]] = disparity_infrared_master[:, :0] * np.ones(
                        [1, disparity_infrared_master.shape[0]])
                else:
                    temp[:cut_vis[0], :cut_vis[1]] = disparity_infrared_master
                    temp[:cut_vis[0], cut_vis[1]:] = disparity_infrared_master[:, -1:] * np.ones(
                        [1, disparity_infrared_master.shape[0]])
            new_disparity = temp
            if cut_inf[0] >= 0:
                if cut_inf[1] >= 0:
                    new_disparity = new_disparity[cut_inf[0]:, cut_inf[1]:]

                else:
                    new_disparity = new_disparity[cut_inf[0]:, :cut_inf[1]]
            else:
                if cut_inf[1] >= 0:
                    new_disparity = new_disparity[:cut_inf[0], cut_inf[1]:]
                else:
                    new_disparity = new_disparity[:cut_inf[0], :cut_inf[1]]

            #################### Projection of the infrared image onto the visible image ################
            print(f"\nStep 3.2) Projection of the infrared left onto the visible right image...")
            infrared_projected = \
                reconstruction_from_disparity(im_infrared_aligned, im_visible_slave_aligned, new_disparity,
                                              min_disp=0, max_disp=0, closing_bool=False, verbose=verbose,
                                              median=False, inpainting=True, copycat=False, orientation=0)
            ref = ImageCustom(im_infrared_aligned).BGR()
            fusion_image = (ref * 0.5 + im_visible_slave_aligned * 0.5) / 255
            new_fusion = (ImageCustom(infrared_projected).BGR() * 0.5 + im_visible_slave_aligned * 0.5) / 255
            if new_disparity.max() <= 0:
                m = -abs(new_disparity).max()
            else:
                m = new_disparity.max()
            image_disparity = ImageCustom(new_disparity / m * 255).RGB(colormap='viridis').LAB()
            ref = ref.LAB()
            ref[:, :, 1:] = image_disparity[:, :, 1:]
            ref = ref.BGR() / 255
            cv.imshow('Disparity superimposed on the image', ref)
            cv.imshow('New Infrared image', infrared_projected / 255)
            cv.imshow('Disparity image', image_disparity.BGR())
            cv.imshow('Fusion image', fusion_image)
            cv.imshow('New Fusion image', new_fusion)
            cv.waitKey(0)
        else:
            for t in Time:
                print(f"{t} processing...")
                if clean:
                    for filename in os.listdir(join(folder, t + "/hybrid/new_disparity")):
                        file_path = os.path.join(folder, t, "hybrid", "new_disparity", filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            print('Failed to delete %s. Reason: %s' % (file_path, e))
                    for filename in os.listdir(join(folder, t + "/hybrid/infrared_projected")):
                        file_path = os.path.join(folder, t, "hybrid", "infrared_projected", filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            print('Failed to delete %s. Reason: %s' % (file_path, e))
                for idx, image_name in tqdm(enumerate(os.listdir(join(folder, t + "/hybrid/right")))):
                    k_str = name_generator(idx)
                    im_visible = cv.imread(join(folder, t + "/hybrid/right", image_name), 1)
                    im_infrared = cv.imread(join(folder, t + "/hybrid/left/left" + k_str + ".png"), 0)
                    if mode == 'image':
                        with open(join(folder, t + "/hybrid/disparity_maps/disparity" + k_str), "rb") as f:
                            disparity = np.array(pickle.load(f))
                        with open(join(folder, t + "/visible/Calibration/transform_matrix"), "rb") as f:
                            matrix = pickle.load(f)
                            cut_vis = (int(-matrix['CutZ'] / 2), int(-matrix['CutY'] / 2))
                        with open(join(folder, t + "/hybrid/Calibration/transform_matrix"), "rb") as f:
                            matrix = pickle.load(f)
                            cut_inf = (-matrix['CutZ'], -matrix['CutY'])
                    else:
                        disparity = cv.imread(join(folder, t + "/hybrid/disparity_maps/disparity" + k_str + ".png"), 0) \
                                    / 255 * post_process
                        with open('/home/godeta/PycharmProjects/LYNRED/LynredDataset/'+ t+"/visible/Calibration/transform_matrix", "rb") as f:
                            matrix = pickle.load(f)
                            cut_vis = (int(-matrix['CutZ'] / 2), int(-matrix['CutY'] / 2))
                        with open('/home/godeta/PycharmProjects/LYNRED/LynredDataset/'+ t+ "/hybrid/Calibration/transform_matrix", "rb") as f:
                            matrix = pickle.load(f)
                            cut_inf = (-matrix['CutZ'], -matrix['CutY'])

                    ################ Reprojection of the disparity maps and adjustment of the border #########################
                    disparity_infrared_master = reprojection_disparity(disparity, translation_ratio, verbose=False)
                    temp = np.zeros([480, 640])
                    if cut_vis[0] >= 0:
                        if cut_vis[1] >= 0:
                            temp[cut_vis[0]:, cut_vis[1]:] = disparity_infrared_master
                            temp[cut_vis[0]:, :cut_vis[1]] = disparity_infrared_master[:, :0] * np.ones(
                                [1, disparity_infrared_master.shape[0]])
                        else:
                            temp[cut_vis[0]:, :cut_vis[1]] = disparity_infrared_master
                            temp[cut_vis[0]:, cut_vis[1]:] = disparity_infrared_master[:, -1:] * np.ones(
                                [1, -cut_vis[1]])
                    else:
                        if cut_vis[1] >= 0:
                            temp[:cut_vis[0], cut_vis[1]:] = disparity_infrared_master
                            temp[:cut_vis[0], :cut_vis[1]] = disparity_infrared_master[:, :0] * np.ones(
                                [1, disparity_infrared_master.shape[0]])
                        else:
                            temp[:cut_vis[0], :cut_vis[1]] = disparity_infrared_master
                            temp[:cut_vis[0], cut_vis[1]:] = disparity_infrared_master[:, -1:] * np.ones(
                                [1, disparity_infrared_master.shape[0]])
                    new_disparity = temp
                    if cut_inf[0] >= 0:
                        if cut_inf[1] >= 0:
                            new_disparity = new_disparity[cut_inf[0]:, cut_inf[1]:]

                        else:
                            new_disparity = new_disparity[cut_inf[0]:, :cut_inf[1]]
                    else:
                        if cut_inf[1] >= 0:
                            new_disparity = new_disparity[:cut_inf[0], cut_inf[1]:]
                        else:
                            new_disparity = new_disparity[:cut_inf[0], :cut_inf[1]]
                    #################### Projection of the infrared image onto the visible image ################
                    infrared_projected = reconstruction_from_disparity(im_infrared, im_visible, new_disparity,
                                                                       min_disp=0, max_disp=0, closing_bool=False,
                                                                       verbose=False,
                                                                       median=False, inpainting=True, copycat=False,
                                                                       orientation=0)
                    ##################### Save ###################################################################
                    name = 'left' + k_str + '.png'
                    path_save = join(folder, t + "/hybrid/infrared_projected/" + name)
                    cv.imwrite(path_save, infrared_projected)
                    name = 'disp' + k_str
                    with open(join(folder, t + "/hybrid/new_disparity/" + name), "wb") as f:
                        pickle.dump(new_disparity, f)


def paralax_parameter_estimation(index=-1, Time=None):
    def nothing(x):
        pass
    if index == -1:
        index = random.randint(0, 99)
    if Time is None:
            Time = 'Day'
    #############################################################################################################
    # 3rd Step : Disparity projection & Final projection
    # We need to reproject the computed disparity maps onto the master IR image and then to project the IR image
    # onto the visible slave image
    # The result will be the fusion of infrared master images (left camera) with the slave visible image (right camera)
    #############################################################################################################
    folder = __path_folder__['image']
    k_str = name_generator(index)
    im_visible = cv.imread(join(folder, Time + "/hybrid/right", "right" + k_str + ".png"), 1)
    im_infrared = cv.imread(join(folder, Time + "/hybrid/left/left" + k_str + ".png"), 0)
    # disparity = cv.imread(join(folder, Time + "/hybrid/disparity_maps/disparity" + k_str + ".png"), 0) / 255 * 70
    with open(join(folder, Time + "/hybrid/disparity_maps/disparity" + k_str), "rb") as f:
        disparity = np.array(pickle.load(f))
    with open('/home/godeta/PycharmProjects/LYNRED/LynredDataset/'+ Time +"/visible/Calibration/transform_matrix", "rb") as f:
        matrix = pickle.load(f)
        cut_vis = (int(-matrix['CutZ'] / 2), int(-matrix['CutY'] / 2))
    with open('/home/godeta/PycharmProjects/LYNRED/LynredDataset/'+ Time + "/hybrid/Calibration/transform_matrix", "rb") as f:
        matrix = pickle.load(f)
        cut_inf = (-matrix['CutZ'], -matrix['CutY'])
    m, n = im_infrared.shape
    cv.namedWindow('Fusion', cv.WINDOW_NORMAL)
    cv.resizeWindow('Fusion', n*2, m)
    cv.createTrackbar('Disparity translation', 'Fusion', 250 + 127, 500, nothing)
    cv.createTrackbar('Disparity scaling', 'Fusion', 100, 200, nothing)
    while True:
        # Updating the parameters based on the trackbar positions
        disp_translation = (cv.getTrackbarPos('Disparity translation', 'Fusion') - 250) / 341
        disp_scaling = cv.getTrackbarPos('Disparity scaling', 'Fusion') / 100
        ################ Reprojection of the disparity maps and adjustment of the border #########################
        disparity_infrared_master = reprojection_disparity(disparity, disp_translation, verbose=False)
        temp = np.zeros([480, 640])
        if cut_vis[0] >= 0:
            if cut_vis[1] >= 0:
                temp[cut_vis[0]:, cut_vis[1]:] = disparity_infrared_master
                temp[cut_vis[0]:, :cut_vis[1]] = disparity_infrared_master[:, :0] * np.ones(
                    [1, disparity_infrared_master.shape[0]])
            else:
                temp[cut_vis[0]:, :cut_vis[1]] = disparity_infrared_master
                temp[cut_vis[0]:, cut_vis[1]:] = disparity_infrared_master[:, -1:] * np.ones(
                    [1, -cut_vis[1]])
        else:
            if cut_vis[1] >= 0:
                temp[:cut_vis[0], cut_vis[1]:] = disparity_infrared_master
                temp[:cut_vis[0], :cut_vis[1]] = disparity_infrared_master[:, :0] * np.ones(
                    [1, disparity_infrared_master.shape[0]])
            else:
                temp[:cut_vis[0], :cut_vis[1]] = disparity_infrared_master
                temp[:cut_vis[0], cut_vis[1]:] = disparity_infrared_master[:, -1:] * np.ones(
                    [1, disparity_infrared_master.shape[0]])
        new_disparity = temp
        if cut_inf[0] >= 0:
            if cut_inf[1] >= 0:
                new_disparity = new_disparity[cut_inf[0]:, cut_inf[1]:]

            else:
                new_disparity = new_disparity[cut_inf[0]:, :cut_inf[1]]
        else:
            if cut_inf[1] >= 0:
                new_disparity = new_disparity[:cut_inf[0], cut_inf[1]:]
            else:
                new_disparity = new_disparity[:cut_inf[0], :cut_inf[1]]
        #################### Projection of the infrared image onto the visible image ################
        infrared_projected = reconstruction_from_disparity(im_infrared, im_visible, new_disparity*disp_scaling,
                                                           inpainting=True)
        ref = ImageCustom(im_infrared).LAB()
        color_distance = ImageCustom(new_disparity/new_disparity.max()*255).RGB(colormap='inferno').LAB()
        ref[:, :, 1:] = color_distance[:, :, 1:]
        ref = ref.RGB()
        fus = (ImageCustom(infrared_projected).RGB(colormap='inferno')*0.5 + im_visible*0.5)
        final = ImageCustom(np.hstack([ref, fus])).BGR()
        cv.imshow("Fusion", final)
        if cv.waitKey(1) == 27:
            break
    cv.destroyAllWindows()



if __name__ == '__main__':
    options = argparse.ArgumentParser()
    """
    These arguments set the method and the source for the disparity estimation
    """
    options.add_argument('--step', default=0, type=int, help='0 means all steps, 1, 2, 3 are the other possibilities')
    options.add_argument('--index', default=-1, type=int, help='index of the picture load, -1 means all of them')
    options.add_argument('--time', default=None, help='Either Day or Night, usefull only if an index is set')
    options.add_argument('--new_rand', action='store_true', help='Generate a new serie of picture index')
    options.add_argument('--monocular', action='store_true',
                         help='Use the monocular depth projection instead of the classical stereo')
    options.add_argument('--calib', action='store_true', help='Calibrate the chosen images')
    options.add_argument('--path', default='image', choices=__path_folder__.keys(),
                         help='Either LynredDataset or Video_frame')
    """
    These arguments set the different parameter of the disparity estimation
    Some arguments are called only by a specific method, it wont be used if the method called is not the good one
    """
    options.add_argument('--ratio', default=127 / (214 + 127), type=float,
                         help='Translation ratio (1 means full translation, - 1 inverse translation)')
    """
    These arguments set the parameters for the disparity maps completion or post processing
    """
    options.add_argument('--post_process', default=70, type=int,
                         help='Post process threshold the disparity map and remove the '
                              'outlier')
    options.add_argument('--verbose', action='store_true', help='Show or not the results along the different steps')
    options.add_argument('--clean', action='store_true', help='Clean the directory before to save the disparity maps')
    options.add_argument('--path_video', default=None, help='Argument used by the video script')
    options = options.parse_args()

    post_process = options.post_process
    index = options.index
    Time = options.time
    translation_ratio = options.ratio
    verbose = options.verbose
    clean = options.clean
    step = options.step
    new_random = options.new_rand
    calibrate = options.calib
    monocular = options.monocular
    folder = __path_folder__[options.path]
    path_video = options.path_video
    mode = options.path

    paralax_correction(post_process=post_process, index=index, Time=Time, translation_ratio=translation_ratio,
                       verbose=verbose, clean=clean,
                       step=step, new_random=new_random, calibrate=calibrate, monocular=monocular, folder=folder,
                       path_video=path_video, mode=mode)

    # paralax_parameter_estimation(Time='Day')

