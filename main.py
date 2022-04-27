# from FUSION.tools.mapping_tools import densification_by_interpolation
from FUSION.tools.registration_tools import *
from Stereo_matching import __method_stereo__, __source_stereo__
from Stereo_matching import *
from Stereo_matching.Tools.disparity_tools import disparity_post_process

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
       - semi-global matching: Algorithm working just like a local matching with added ordering constraints and others.
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
   
    Alpha) DataLoader --> provide the data specified by "source"
        If source is a Dataset name, the dataloader will prepare the computation of the whole dataset.
    
   0) Image Pre-processing (if needed):
       - Image edges extraction
       - Change of color-space
       - Computation of the scale pyramid... etc.

   1) Computation of the disparity map
       - Some algorithm give a dense map, some other just a scarce one. 
       We need to densify the scarce estimation in order to use it properly.
       --> Densifyition by interpolation
   :param left: left image.
   :param right: right image.
   :param parameters: structure containing parameters of the algorithm.
   :param save_images: whether to save census images or not.
   :return: H x W x D array with the matching costs.
   """

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    """
    These arguments set the method and the source for the disparity estimation
    """
    parser.add_argument('--method', default='SBGM', choices=__method_stereo__.keys(), help='Name of the method or the Net used')
    parser.add_argument('--source', default='cones', choices=__source_stereo__.keys(), help='Either, teddy, cones or lynred')
    parser.add_argument('--calib', action='store_true', help='Use the calibration images from the Lynred dataset')

    """
    These arguments set the different parameter of the disparity estimation
    Some arguments are called only by a specific method, it wont be used if the method called is not the good one
    """
    parser.add_argument('--min_disp', default=20, type=int, help='minimum disparity for the stereo pair')
    parser.add_argument('--max_disp', default=192, type=int, help='maximum disparity for the stereo pair')
    """
    These arguments set the parameters for the disparity maps completion or post processing
    """
    parser.add_argument('--scale', default=0, type=int, help='upscale the disparity map for reconstruction')
    parser.add_argument('--closing', default=0, type=int,
                        help='Apply or not a closing operator on the result, enter the footprint')
    parser.add_argument('--median', default=0, type=int,
                        help='Apply or not a median filter on the result, enter the footprint')
    parser.add_argument('--edges', action='store_true', help='Conserve the edges, displace only the texture')
    parser.add_argument('--inpainting', action='store_true',
                       help='Use the Inpainting function to fill the gap in the new image')
    parser.add_argument('--copycat', action='store_true',
                       help='Use the second image to fill the gap in the new image')
    parser.add_argument('--dense', action='store_true', help='Add a densification step for the disparity map')
    parser.add_argument('--post_process', default=0, type=int, help='Post process threshold the disparity map and remove the '
                                                                    'outlier')
    """
    These argument show the result of the operations along the way
    """
    parser.add_argument('--verbose', action='store_true', help='Show or not the results along the different steps')
    # parser.add_argument('--edges', default=False, action=argparse.BooleanOptionalAction,
    #                     help='Conserve the edges, displace only the texture')
    # parser.add_argument('--inpainting', default=False, action=argparse.BooleanOptionalAction,
    #                     help='Use the Inpainting function to fill the gap in the new image')
    # parser.add_argument('--dense', default=False, action=argparse.BooleanOptionalAction,
    #                     help='Add a densification step for the disparity map')
    # parser.add_argument('--binary', default=False, action=argparse.BooleanOptionalAction,
    #                     help='Classical SGM when False, SGBM when True')
    # parser.add_argument('--binary', action='store_true', help='Classical SGM when False, SGBM when True')
    # parser.add_argument('--verbose', default=False, action=argparse.BooleanOptionalAction,
    #                     help='Show or not the result')

    args = parser.parse_args()

    calib = args.calib
    method = args.method
    source = args.source
    min_disp = args.min_disp
    max_disp = args.max_disp
    verbose = args.verbose
    scale = args.scale
    closing_bool = args.closing
    median = args.median
    edges = args.edges
    inpainting = args.inpainting
    copycat = args.copycat
    dense = args.dense
    post_process = args.post_process

    start = time.time()
    '''
        Alpha-STEP: loading of the data according the source chosen
        The "sample" returned is a dictionary with the keys "imgL" and "imgR"
        The images are float array with values between 0 and 1
    '''
    sample = dataloader(__source_stereo__[source], source, calib=calib)
    '''
        1st-STEP: computation of the Disparity Map using the selected method
        SGM & SGBM :
         - input : Source name (drive, teddy or cones), min_disp (disparity min in pixels, not required), 
                   binary (for the use of SGBM, by default), verbose : To show and set the different parameters
    '''
    print(f"\n1) Computation of the disparity map...")
    if method == 'SGM':
        from Stereo_matching.Algorithms.SGM.OpenCv_DepthMap.depthMapping import depthMapping
        imageL, imageR, disparity_matrix, m, M = depthMapping(sample, source, min_disp, max_disp, 0, verbose, edges)
    if method == 'SBGM':
        from Stereo_matching.Algorithms.SGM.OpenCv_DepthMap.depthMapping import depthMapping
        imageL, imageR, disparity_matrix, m, M = depthMapping(sample, source, min_disp, max_disp, 1, verbose, edges)
    elif method == 'MobileStereoNet':
        from Stereo_matching.NeuralNetwork.MobileStereoNet.image_depth_estimation import mobilestereonet
        imageL, imageR, disparity_matrix, m, M = mobilestereonet(sample, verbose=verbose)
    elif method == 'ACVNet':
        from Stereo_matching.NeuralNetwork.ACVNet_main.ACVNet_test import ACVNet_test
        imageL, imageR, disparity_matrix, m, M = ACVNet_test(sample, max_disp, verbose)
        imageL, imageR = np.uint8(imageL*255), np.uint8(imageR*255)

    '''
        2nd-STEP: Densification and amelioration of the Disparity Map (optionnal) :
    '''
    if dense:
        print(f"\n2) Densification of the disparity map...")
        met = ["CloughTorcher", 'inpainting', 'griddata', "interpolation"]
        # maps = densification_by_interpolation(maps, method=met[3], verbose=verbose)
    else:
        print(f"\n2) No Densification...")
    if post_process:
        print(f"    Post processing of the disparity map...")
        disparity_matrix = disparity_post_process(disparity_matrix, m, M, post_process)
    '''
        3rd-STEP: Reconstruction from Disparity Map :
    '''
    print(f"\n3) Reconstruction of the image from the disparity map...")
    image_corrected = reconstruction_from_disparity(imageL, imageR, disparity_matrix, m, M, scale, closing_bool, verbose, median,
                                                    inpainting, copycat, orientation=0)
    '''
        4th-STEP: Error Estimation :
        The following Errors are implemented : L1 - SSIM - RMSE
    '''
    print(f"\n4) Computation of the different indexes...")
    # if method == "Net":
    #     if len(imageL.shape) > 2:
    #         ref = cv.cvtColor(imageL, cv.COLOR_BGR2GRAY)
    #         image_corrected_gray = cv.cvtColor(image_corrected, cv.COLOR_BGR2GRAY)
    #     else:
    #         ref = imageL
    # else:
    if len(imageR.shape) > 2:
        ref = cv.cvtColor(imageR, cv.COLOR_BGR2GRAY)
        image_corrected_gray = cv.cvtColor(image_corrected.astype(np.uint8), cv.COLOR_BGR2GRAY)
    else:
        ref = imageR
        image_corrected_gray = image_corrected
    error_estimation(image_corrected_gray, ref, ignore_undefined=True)
    cv.imshow('Reconstruct image', image_corrected / 255)
    cv.imshow('Difference Reconstruct left Right', abs(image_corrected_gray / 255 - ref / 255))
    plt.matshow(disparity_matrix)

    print(f"Total time : {time.time() - start} seconds !")
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()

