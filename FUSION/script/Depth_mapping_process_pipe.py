import random
import os
from os.path import *
import tkinter as tk

import lynred_py
from cv2 import waitKey
from imutils.video import VideoStream
from FUSION.classes.Camera import Camera
from FUSION.tools import gradient_tools
from FUSION.tools.data_management_tools import register_cmap, open_image
from FUSION.tools.manipulation_tools import *
# from FUSION.tools.method_fusion import colormap_fusion
from FUSION.tools.gradient_tools import *
from FUSION.tools.method_fusion import *
from FUSION.tools.registration_tools import *
import numpy as np
from FUSION.interface.Application import Application
import time
from scipy.ndimage import median_filter


class Depth_map_pipe:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'pre_processing':
                self.preProcessing = value
            elif key == 'disparity' or key == 'disparity_method':
                self.disparity_method = value
            elif key == 'fusion_method':
                self.fusion_method = value
            elif key == 'error_method':
                self.error_method = value
        self.disparity_map = None

    def apply(self, image_ir, image_vis):
        image_ir, image_vis = self.apply_processing(image_ir, image_vis)
        self.disparity_map = self.compute_depth_map(image_ir, image_vis)
        image_ir = self.correct_disparity(image_ir)

    def apply_processing(self, image_ir, image_vis):
        for function in self.preProcessing:
            image_ir = function(image_ir)
            image_vis = function(image_vis)
        return image_ir, image_vis

    def compute_disparity(self, image_ir, image_vis):
        if self.fusion_method == "pixel":
            return None

    """
    #################################################################################
    SUMMARY:
    The main goal here is to somehow associate as much information possible from the 
    infrared image to the visible image taking into account the stereo effect.
    We gonna review different technique from the literature working with stereo pair
    of visible images and try to adapt them as best as we can. 
    We will also propose some methods of our own using template matching with 
    common features as the gradient or the phase.
    #############################################
    LITERATURE:
    We will have here an overview of the different technics used for 'classical' stereo 
    matching using a stereo visible database of images. That search can be split in two part :
    - The 'conventional algorithm', they can be "global", "semi-global" or local 
    - The NN based techniques witches mostly make use of CNN
    
    Algorithms:
        - global matching: minimize the global energy functions : E(d) = Edata(d) + Esmooth(d)
            ex: graph cut, belief propagation
        - semi-global matching: 
        - local_matching: Try to find the best correlation of a part of the image in the other 
            by a sliding window search. Our method is based on this kind of algorithm.
            ex: SSD algorithm
    #############################################
    GET THE CORRECTION FROM THE DISPARITY MAP:
    If we consider the following spatial offset between the cameras : x1= R.x0 + t
    then we have the essential matrix:
        |0 -tz -ty|
    E = |tz  0 -tx|
        |-ty tx  0|
    which maps a point x0 from the left image to a line l1=Ex0 in the right image.
    If the images are rectified, thus the rotation is null, there is only a translation 
    between the two cameras over the x axis, this relation becomes : d=Bf/Z  
    with f the focal length (in pixels) , B the baseline
    Presentation of the pipe of processing:
    
    1) Image Pre-processing (if needed):
        - Image edges extraction
        - Change of color-space
        - Computation of the scale pyramid... etc.
        
    2) Computation of the disparity map
        - 
    :param left: left image.
    :param right: right image.
    :param parameters: structure containing parameters of the algorithm.
    :param save_images: whether to save census images or not.
    :return: H x W x D array with the matching costs.
    """


ir, vis = open_image()
