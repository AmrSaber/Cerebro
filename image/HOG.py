from skimage import io
from skimage.feature import hog
#import matplotlib.pyplot as plt


def getHOG(img):
    features, hog_image = hog(img,
          orientations=8,
          pixels_per_cell=(12, 12),
          cells_per_block=(4, 4),
          visualize=True,
          transform_sqrt=False,
          feature_vector=True,
          multichannel=True)

    #to display

    #io.imshow(hog_image)
    #plt.show()
    
    return features

