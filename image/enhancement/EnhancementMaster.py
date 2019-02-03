import numpy as np
import cv2


class EnhancementMaster:
	def __init__(self):
		raise Exception("don't make an object from this class")

	@staticmethod
	def apply_equalization(img):
		output = img.copy()
		for i in (0, 1, 2):
			output[:, :, i] = cv2.equalizeHist(img[:, :, i])
		# img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

		# equalize the histogram of the Y channel
		# img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

		# convert the YUV image back to RGB format
		# output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
		return output

	@staticmethod
	def apply_sharpening(img):
		# img = cv2.GaussianBlur(img, (5, 5), 0)
		# smooth = cv2.addWeighted(blur, 1.5, img, -0.5, 0)
		kernel = np.array([
				[0, -1, 0],
				[-1, 5, -1],
				[0, -1, 0]
		])
		return cv2.filter2D(img, -1, kernel)

	@staticmethod
	def apply_median(img):
		return cv2.medianBlur(img)


if __name__ == '__main__':
	im = cv2.imread('test.jpeg')
	cv2.imshow("original", im)
	cv2.waitKey(0)
	cv2.imshow("Equalized", EnhancementMaster.apply_equalization(im))
	cv2.waitKey(0)
	cv2.imshow("Sharpened", EnhancementMaster.apply_sharpening(im))
	cv2.waitKey(0)

