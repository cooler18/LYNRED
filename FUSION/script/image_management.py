from FUSION.tools.data_management_tools import *
from matplotlib.colors import ListedColormap
import matplotlib.cm
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

## Code for automatic images registration

from FUSION.tools.registration_tools import manual_registration, automatic_registration

verbose = False
p = "/home/godeta/PycharmProjects/LYNRED/Images/Day"
p_save = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/visible/Day"
with open("/home/godeta/PycharmProjects/LYNRED/LynredDataset/visible/Day/Calibration/transform_matrix_slaveToMaster_vis", "rb") as f:
    matrix = pickle.load(f)
list_left = sorted(os.listdir(join(p, 'master', 'visible')))
list_right = sorted(os.listdir(join(p, 'slave', 'visible')))
for i in range(20):
    n = np.random.randint(0, len(os.listdir(join(p, 'slave', 'visible')))-1, 1)[0]
    imgR = join(p, 'slave', 'visible', list_right[n])
    imgL = join(p, 'master', 'visible', list_left[n])
    imgL = ImageCustom(imgL)
    imgR = ImageCustom(imgR)
    if verbose:
        cv.imshow('control Left', imgL.BGR())
        cv.imshow('control Right', imgR.BGR())
        cv.waitKey(0)
        cv.destroyAllWindows()
    n = len(os.listdir(join(p_save, 'left')))
    nameL = 'left'+str(n)+'.png'
    nameR = 'right' + str(n) + '.png'
    automatic_registration(imgL, imgR, matrix, nameL=nameL, nameR=nameR)
print(f"Automatic registration of the {i+1} images is done !")

imgL = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/visible/Day/Calibration/Calib_left.png"
imgR = "/home/godeta/PycharmProjects/LYNRED/LynredDataset/visible/Day/Calibration/Calib_right.png"
imgL = ImageCustom(imgL)
imgR = ImageCustom(imgR)
manual_registration(imgL, imgR)



