#!/usr/bin/env python3

import cv2

from interface import video_stream as vs
from interface import process_image

def main():
	fps = 40
	pa = argparse.ArgumentParser()
	pa.add_argument('-s', action='store_true', help='stream from camera')
	args = pa.parse_args()
	if args.s:
		vs.detect_stream_emotions(fps)
if __name__ == '__main__': main()
