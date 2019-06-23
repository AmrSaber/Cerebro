#! /usr/bin/env python3

import numpy as np
import cv2
# import matplotlib.pyplot as plt

import image.face_detector.detect_dlib as dlib_detector
import image.face_detector.detect_haar as haar_detector
import image.face_detector.detect_lbp as lbp_detector

def get_squared(img, n):
    ret = img.copy()
    width = height = n # np.max((img.shape[0], img.shape[1]))
    ret = cv2.resize(img, (width, height),interpolation = cv2.INTER_AREA)
    return ret

def extend(img):
    ret = img.copy()
    if img.shape[0] < img.shape[1]:
        diff = img.shape[1] - img.shape[0]
        add = np.array([[[0] * 3] * img.shape[1]] * diff)
        ret = np.vstack((add, img))
    if img.shape[0] > img.shape[1]:
        diff = img.shape[0] - img.shape[1]
        add = np.array([[[0] * 3] * diff] * img.shape[0])
        ret = np.hstack((add, img))
    return ret

def normalize_image(img, n, detect=False, detector='dlib'):
    ret = img.copy()

    # apply face detection if asked
    if detect:
        if detector == 'dlib':
            ret = dlib_detector.get_faces(img)[0][0]
        elif detector == 'haar':
            ret = haar_detector.get_faces(img)[0][0]
        elif detector == 'lbp':
            ret = lbp_detector.get_faces(img)[0][0]
        else:
            raise Exception('unknown detector %s given' %detector)

    # making the image a square image by adding padding depending on the shape
    if ret.shape[0] < ret.shape[1]:
        diff = ret.shape[1] - ret.shape[0]
        ret = cv2.copyMakeBorder(ret,diff,0,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
    elif ret.shape[0] > ret.shape[1]:
        diff = ret.shape[0] - ret.shape[1]
        ret = cv2.copyMakeBorder(ret,0,diff,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])

    # resize image to given size
    if ret.shape[0] != n:
        ret = cv2.resize(ret, (n, n))

    return ret

def normalize_channels(image):
    if len(image.shape) == 3 and image.shape[-1] == 3:
        ret = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        ret = image.copy()
    return ret
