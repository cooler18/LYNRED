import os
import pickle
from os.path import join, dirname, abspath
import shutil
from tkinter.filedialog import askopenfilename

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from skimage import io
import cv2 as cv

from FUSION.classes.Image import ImageCustom
from FUSION.interface.interface_tools import search_number, search_ext, random_image_opening


def copy_image(DIR, DEST_DIR, init=False):
    if init:
        for folder in os.listdir(DEST_DIR):
            try:
                shutil.rmtree(join(DEST_DIR, folder))
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
        os.mkdir(join(DEST_DIR, "infrared"))
        os.mkdir(join(DEST_DIR, "visible"))
        os.mkdir(join(DEST_DIR, "multispectral"))
    num = len(os.listdir(DEST_DIR + "/visible"))

    for folder in os.listdir(DIR):
        if num > 1000:
            break
        path = DIR + "/" + folder
        if folder == "infrared" or folder == 'visible' or folder == 'multispectral':
            print(f" Le dossier {folder} est en cours de copie dans {path}....")
            for image in os.listdir(path):
                if folder == "infrared":
                    ir = len(os.listdir(DEST_DIR + "/infrared"))
                    filename, file_extension = os.path.splitext(image)
                    target = DEST_DIR + "/infrared/IFR_" + str(ir) + file_extension
                elif folder == 'visible':
                    vis = len(os.listdir(DEST_DIR + "/visible"))
                    filename, file_extension = os.path.splitext(image)
                    target = DEST_DIR + "/visible/VIS_" + str(vis) + file_extension
                else:
                    mul = len(os.listdir(DEST_DIR + "/multispectral"))
                    filename, file_extension = os.path.splitext(image)
                    target = DEST_DIR + "/multispectral/MUL_" + str(mul) + file_extension
                source = path + "/" + image
                shutil.copy(source, target)
            num = len(os.listdir(DEST_DIR + "/visible"))
            print(f"{num} images copiées")
        elif folder[-4] == "." or folder[-5] == ".":
            pass
        elif folder[-6:] == "winter":
            pass
        else:
            copy_image(path, DEST_DIR)


cmaps = {}

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    # axs[0].set_title(f'{category} colormaps', fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()
    plt.show()

    # Save colormap list for later.
    cmaps[category] = cmap_list


def concatane_train_label(SRC_DIR, n=-1):
    train_path = SRC_DIR + "/infrared"
    label_path = SRC_DIR + "/visible"
    if n > 0:
        shape = io.imread(join(train_path, os.listdir(train_path)[0])).shape
        train = np.empty([n, shape[0], shape[1], 1])
        label = np.empty([n, shape[0], shape[1], 3])
        for im in range(n):
            train[im, :, :, 0] = io.imread(join(train_path, os.listdir(train_path)[im]))[:, :, 0]
            label[im, :, :, :] = cv.resize(io.imread(join(label_path, os.listdir(label_path)[im])),
                                           (shape[1], shape[0]),
                                           interpolation=cv.INTER_AREA)
    else:
        shape = io.imread(join(train_path, os.listdir(train_path)[0])).shape
        train = np.empty([n, shape[0], shape[1], 1])
        label = np.empty([n, shape[0], shape[1], 3])
        for im in range(len(os.listdir(train_path))):
            train[im, :, :, 0] = io.imread(join(train_path, os.listdir(train_path)[im]))[:, :, 0]
            label[im, :, :, :] = cv.resize(io.imread(join(label_path, os.listdir(label_path)[im])),
                                           (shape[0], shape[1]),
                                           interpolation=cv.INTER_AREA)
    with open(join(SRC_DIR, "train.npy"), "wb") as p:
        pickle.dump(train, p)
    with open(join(SRC_DIR, "label.npy"), "wb") as p:
        pickle.dump(label, p)


def register_cmap_Lynred(path):
    with open(path) as f:
        lines = f.readlines()
    name = lines[0].split()[1]
    cmap = np.zeros([256, 4])
    for l in lines[4:]:
        idx, r, g, b, a = int(l.split()[0]), float(l.split()[1]) / 255, float(l.split()[2]) / 255, float(
            l.split()[3]) / 255, 1
        cmap[idx] = r, g, b, a
    Lynred_cmap = ListedColormap(cmap, name=name)
    matplotlib.cm.register_cmap(name=name, cmap=Lynred_cmap)


def register_cmap(cmap, name):
    c_map = ListedColormap(cmap, name=name)
    matplotlib.cm.register_cmap(name=name, cmap=c_map)


def open_image():
    p = dirname(abspath('.'))
    filename = askopenfilename(title="Open an image", filetypes=[('jpg files', '.jpg'), ('png files', '.png'),
                                                                 ('all files', '.*')],
                               initialdir=join(p, 'Images_grouped', 'visible'))
    num = search_number(filename)
    ext = search_ext(join(p, 'Images_grouped', 'infrared'), num)
    filename_ir = p + "/Images_grouped/infrared/IFR_" + num + ext
    VIS = ImageCustom(filename)
    IR = ImageCustom(filename_ir)
    return IR, VIS


def open_image_random():
    vis, ir, _ = random_image_opening()
    VIS = ImageCustom(vis)
    IR = ImageCustom(ir)
    return IR, VIS


def open_image_manually():
    p = dirname(abspath('.'))
    filename_vis = askopenfilename(title="Open an image Visible",
                                   filetypes=[('jpg files', '.jpg'), ('png files', '.png'),
                                              ('all files', '.*')],
                                   initialdir=join(p, 'Images_grouped', 'visible'))
    filename_ir = askopenfilename(title="Open an image Infrared",
                                  filetypes=[('tiff files', '.tiff'), ('png files', '.png'),
                                             ('all files', '.*')],
                                  initialdir=join(p, 'Images_grouped', 'infrared'))
    VIS = ImageCustom(filename_vis)
    IR = ImageCustom(filename_ir)
    return IR, VIS
