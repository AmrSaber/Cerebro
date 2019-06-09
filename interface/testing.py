#!/usr/bin/env python3

from interface import video_stream as vs

def main():
	 fps = 40
	 vs.detect_stream_emotions(fps)

if __name__ == '__main__': main()
