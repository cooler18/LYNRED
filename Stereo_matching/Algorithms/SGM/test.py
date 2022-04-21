import cv2 as cv
from FUSION.classes.Image import ImageCustom


im_rgb = ImageCustom("D:\Travail\LYNRED\Stereo_matching\Algorithms\SGM\semi-global-matching\Samples\VIS_32.jpg")
im_ir = ImageCustom("D:\Travail\LYNRED\Stereo_matching\Algorithms\SGM\semi-global-matching\Samples\IFR_32.tiff")

im_rgb = cv.resize(im_rgb, (im_ir.shape[1], im_ir.shape[0]))

im_rgb = cv.cvtColor(im_rgb, cv.COLOR_RGB2BGR)
cv.imwrite("D:\Travail\LYNRED\Stereo_matching\Algorithms\SGM\semi-global-matching\Samples\left_drive.png", im_ir)
cv.imwrite("D:\Travail\LYNRED\Stereo_matching\Algorithms\SGM\semi-global-matching\Samples/right_drive.png", im_rgb)
