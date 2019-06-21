#! /user/bin/env python3

import numpy as np
import cv2

def laplacian(img):
    laplacian = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, laplacian)

# for removing salt and peper
def median(img, size=3):
    return cv2.medianBlur(img, size)

# for removing gausian
def fastNLMeans(img, h=15, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoising(img, None, h, templateWindowSize, searchWindowSize)
