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
with open('D:\Travail\LYNRED\FUSION\interface\data_interface\LUT8_lifeinred_color.dat') as f:
    lines = f.readlines()
name = lines[0].split()[1]
cmap = np.zeros([256, 4])
for l in lines[4:]:
    idx, r, g, b, a = int(l.split()[0]), float(l.split()[1])/255, float(l.split()[2])/255, float(l.split()[3])/255, 1
    cmap[idx] = r, g, b, a
Lynred_cmap = ListedColormap(cmap, name=name)
matplotlib.cm.register_cmap(name=name, cmap=Lynred_cmap)
plot_color_gradients('Perceptually Uniform Sequential',
                     ['lifeinred_color'])#, 'plasma', 'inferno', 'magma', 'cividis'])

#######################################################################
# Script to generate the train and label for RGB coloration of IR image by CNN
# SRC_DIR = "D:\Travail\LYNRED\FUSION\Images_grouped"
# concatane_train_label(SRC_DIR, n=20)

#########################################################################
