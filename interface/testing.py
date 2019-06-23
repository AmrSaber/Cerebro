#!/usr/bin/env python3

from interface import video_stream as vs

def main():
	vs.detect_stream_emotions()
	# imagePathes = [
	# 	'./samples/baby_sad.jpg'
	# 	'./samples/baby_sad2.jpg'	
	# 	# './samples/people_sad.jpg',
	# 	# './samples/people_happy.jpeg'
	# ]

	# for imagePath in imagePathes:
	# 	image = cv2.imread(imagePath)
	# 	markedImage = process_image.mark_faces_emotions(image)
	# 	cv2.imshow('Marked Face(s)', markedImage)
	# 	cv2.waitKey()

	# fps = 70
	# pa = argparse.ArgumentParser()
	# pa.add_argument('-c', action='store_true', help='stream from camera')
	# args = pa.parse_args()
	# if args.c:
	# 	vs.detect_stream_emotions(fps)

	pv.detect_video_emotions_with_tracking('interface/y.mp4', 'interface/y_output.mp4')

if __name__ == '__main__': main()
