#!/usr/bin/env python3

import cv2

from image import utils

#from interface import process_video as pv
from interface import process_image
from interface import video_stream as vs

def main():
	 fps = 40
	 vs.detect_stream_emotions(fps)
'''
	image = cv2.imread('./interface/1.jpg')
	result = process_image.mark_faces_emotions(image)
	
	cv2.imshow("image", result)
	cv2.waitKey(0)
'''
	# pa = argparse.ArgumentParser()
	# pa.add_argument('-s', action='store_true', help='stream from camera')
	# args = pa.parse_args()
	# if args.s: pass

if __name__ == '__main__': main()
