import argparse
import video_stream as vs

def main():
	fps = 40
	pa = argparse.ArgumentParser()
	pa.add_argument('-s', action='store_true', help='stream from camera')
	args = pa.parse_args()
	if args.s:
		vs.detect_stream_emotions(fps)
if __name__ == '__main__': main()
