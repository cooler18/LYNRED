import tkinter as tk
from tkinter.filedialog import *
from PIL import ImageTk, Image
import PIL
import random
from os.path import *
import skimage.io as io
import cv2 as cv
import os


def prepare_image(image, size=(640, 480), master=None):
    if image.shape[0] != size[1] or image.shape[1] != size[0]:
        image = cv.resize(image, size, interpolation=cv.INTER_AREA)
    image_PIL = Image.fromarray(image)  # Conversion du format RVB au format PIL
    if master:
        image_tk = ImageTk.PhotoImage(image_PIL, master=master)  # Convertir au format ImageTk
    else:
        image_tk = ImageTk.PhotoImage(image_PIL)  # Convertir au format ImageTk
    return image_tk


def highlight(widget):
    widget['bg'] = 'red'
    widget['relief'] = 'sunken'


def random_image_opening(verbose=1):
    p = dirname(abspath('FUSION'))
    pathrgb = p + "/../Images_grouped/visible"
    pathgray = p + "/../Images_grouped/infrared"
    pathfus = p + "/../Images_grouped/multispectral"
    random.seed()
    n = random.randint(0, len(os.listdir(pathrgb)) - 1)
    ext_rgb = search_ext(pathrgb, n)
    ext_ir = search_ext(pathgray, n)
    ext_fus = search_ext(pathfus, n)
    imageRGB_name = pathrgb + "/VIS_" + str(n) + ext_rgb
    imageIR_name = pathgray + "/IFR_" + str(n) + ext_ir
    imageFUS_name = pathfus + "/MUL_" + str(n) + ext_fus
    if verbose:
        print(f"Image number : {n}")
    return imageRGB_name, imageIR_name, imageFUS_name


def search_number(path):
    name = basename(path)
    name = name[4:]
    c = name[0]
    i = 0
    j = 0
    while c != '.':
        i += 1
        c = name[i]
    return name[:i]


def search_ext(path, num):
    p = ""
    for image in os.listdir(path):
        p = join(path, image)
        num_comp = search_number(p)
        if num_comp != num:
            pass
        else:
            break
    filename, file_extension = os.path.splitext(p)
    return file_extension


def clearImage(canva):
    canva.delete("all")


def disableChildren(parent):
    for child in parent.winfo_children():
        wtype = child.winfo_class()
        if wtype not in ('Frame', 'Labelframe'):
            child.configure(state='disable')
        else:
            disableChildren(child)


def enableChildren(parent):
    for child in parent.winfo_children():
        wtype = child.winfo_class()
        if wtype not in ('Frame', 'Labelframe'):
            child.configure(state='normal')
        else:
            enableChildren(child)
