from skimage import io
from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2

def sk_get_hog(img):
	features, hog_image = hog(img,
		orientations=9,
		pixels_per_cell=(8, 8),
		cells_per_block=(3, 3),
		visualize=True,
		transform_sqrt=False,
		feature_vector=True,
		multichannel=True)
	io.imshow(hog_image)
	plt.show()
	return features

def cv_get_hog(img):
	hog = cv2.HOGDescriptor()
	features = hog.compute(img)
	return features
