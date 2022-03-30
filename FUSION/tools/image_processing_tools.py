import time
import cv2 as cv
from FUSION.classes.Image import ImageCustom


def histogram_equalization(image, method=0):
    if method == 0:
        return image
    elif method == 1:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if image.cmap == 'GRAYSCALE':
            # start = time.time()
            image = ImageCustom(clahe.apply(image), image)
            # check1 = time.time() - start
            # print(f"1st step : {check1} sec")
        elif image.cmap == 'RGB' or image.cmap == 'BGR':
            # start = time.time()
            image = image.LAB()
            # check1 = time.time() - start
            # start = time.time()
            image[:, :, 0] = clahe.apply(image[:, :, 0])
            # check2 = time.time() - start
            # start = time.time()
            image = image.BGR()
            # check3 = time.time() - start
            # print(f"1st step : {check1} sec\n"
            #       f"2nd step : {check2} sec\n"
            #       f"3rd step : {check3} sec")
        return image
    elif method == 2:
        if image.cmap == 'GRAYSCALE':
            image = ImageCustom(cv.equalizeHist(image), image)
        elif image.cmap == 'RGB' or image.cmap == 'BGR':
            image = image.LAB()
            image[:, :, 0] = cv.equalizeHist(image[:, :, 0])
            image = image.BGR()
        return image
