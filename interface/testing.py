import argparse
import video_stream as vs

def main():
	fps = 70
	pa = argparse.ArgumentParser()
	pa.add_argument('-c', action='store_true', help='stream from camera')
	args = pa.parse_args()
	if args.c:
		vs.detect_stream_emotions(fps)
if __name__ == '__main__': main()
