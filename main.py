from FUSION.tools.mapping_tools import densification_by_interpolation
from FUSION.tools.registration_tools import *

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
    parser.add_argument('--method', default='SGM', help='Name of the method : SGM')
    parser.add_argument('--source', default='drive', help='Either, teddy, cones or drive')
    parser.add_argument('--disp', default=20, type=int, help='minimum disparity for the stereo pair')
    parser.add_argument('--binary', default=False, action=argparse.BooleanOptionalAction,
                        help='Classical SGM when False, SGBM when True')
    parser.add_argument('--verbose', default=False, action=argparse.BooleanOptionalAction,
                        help='Show or not the result')
    parser.add_argument('--scale', default=0, type=int, help='Either, teddy, cones or drive')
    parser.add_argument('--closing', default=0, type=int,
                        help='Apply or not a closing operator on the result, enter the footprint')
    parser.add_argument('--median', default=0, type=int,
                        help='Apply or not a median filter on the result, enter the footprint')
    parser.add_argument('--edges', default=False, action=argparse.BooleanOptionalAction,
                        help='Conserve the edges, displace only the texture')
    parser.add_argument('--inpainting', default=False, action=argparse.BooleanOptionalAction,
                        help='Use the Inpainting function to fill the gap in the new image')
    parser.add_argument('--dense', default=False, action=argparse.BooleanOptionalAction,
                        help='Add a densification step for the disparity map')
    args = parser.parse_args()

    method = args.method
    source = args.source
    min_disp = args.disp
    binary = args.binary
    verbose = args.verbose
    scale = args.scale
    closing_bool = args.closing
    median = args.median
    edges = args.edges
    inpainting = args.inpainting
    dense = args.dense

    '''
        1st-STEP: computation of the Disparity Map using the selected method
        SGM & SGBM :
         - input : Source name (drive, teddy or cones), min_disp (disparity min in pixels, not required), 
                   binary (for the use of SGBM, by default), verbose : To show and set the different parameters
    '''
    print(f"\n1) Computation of the disparity map...")
    if method == 'SGM':
        from Stereo_matching.Algorithms.SGM.OpenCv_DepthMap.depthMapping import depthMapping
        imageL, imageR, maps, m, M = depthMapping(source, min_disp, binary, verbose, edges)
    elif method == 'Net':
        from Stereo_matching.NeuralNetwork.MobileStereoNet.image_depth_estimation import mobilestereonet
        imageL, imageR, maps, m, M = mobilestereonet(source, verbose=verbose)

    '''
        2nd-STEP: Densification of the Disparity Map (optionnal) :
    '''
    if dense:
        print(f"\n2) Densification of the disparity map...")
        met = ["CloughTorcher", 'inpainting', 'griddata', "interpolation"]
        maps = densification_by_interpolation(maps, method=met[3], verbose=verbose)
    else:
        print(f"\n2) No Densification...")

    '''
        3rd-STEP: Reconstruction from Disparity Map :
    '''
    print(f"\n3) Reconstruction of the image from the disparity map...")
    image_corrected = reconstruction_from_disparity(imageL, imageR, maps, m, M, scale, closing_bool, verbose, median,
                                                    inpainting, orientation=0)

    '''
        4th-STEP: Error Estimation :
        The following Errors are implemented : L1 - SSIM - RMSE
    '''
    print(f"\n4) Computation of the different indexes...")
    if method == "Net":
        if len(imageL.shape) > 2:
            ref = cv.cvtColor(imageL, cv.COLOR_BGR2GRAY)
        else:
            ref = imageL
    else:
        if len(imageR.shape) > 2:
            ref = cv.cvtColor(imageR, cv.COLOR_BGR2GRAY)
        else:
            ref = imageR
    error_estimation(image_corrected, ref, ignore_undefined=True)
    cv.imshow('Reconstruct image', image_corrected / 255)
    cv.imshow('Difference Reconstruct left Right', image_corrected / 255 - ref / 255)
    plt.matshow(maps)
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()
