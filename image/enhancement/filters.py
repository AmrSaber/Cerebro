#! /user/bin/env python3

import numpy as np
import cv2

def laplacian(img):
    laplacian = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, laplacian)

def median(img, size=3):
    return cv2.medianBlur(img, size)
