# Cerebro
Cerebro is a python package for **Facial Expression Detection**, we provide a trained model with accuracy around 98% of 8 emotions [Happy, surprise, contempt, sad ,angry,disgust,Neutral,Fear], with a very simple interface for detection from image,video with any rotation and Real time streaming.

**Documentation**

**Example**
In this Example we get an image , predict an emotion then save it with the emotion.
```python
from interface import video_stream as vs
from interface import process_image as pi

def main():
	im = cv2.imread("interface/7.jpg")
	items =pi.extract_faces_emotions(im)
	im =pi.mark_faces_emotions(im)
	cv2.imwrite("interface/77.jpg",im)
	cv2.imshow("detected emotions",im)
	cv2.waitKey(0)
    
if __name__ == '__main__': main()

```
![alt text](https://github.com/AmrSaber/Cerebro/blob/master/images/BeFunky-collage.jpg "Example")

**Installition**
Cerebro depends on some python packages, once you install Cerebro any missing Module will be automatically installed, 
for ***FFmpeg*** use this [link](https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg).

***Installation by hand***: download the sources, either from [PyPI](https://test.pypi.org/project/CEREBRO1/#description) or, if you want the development version, from GitHub, clone the project then use this command in terminal to setup.

```$ (sudo) python setup.py install```

***Installation with pip***: if you have pip installed, just type this in a terminal:

```$ (sudo) pip install Cerebro```
***Using Model*** : once You install Cerebro You have to dowenload our trained model from this [link](https://github.com/AmrSaber/Cerebro/blob/master/Cerebro/saved-models/emotions_model_specs.bin) and full model [link](https://github.com/AmrSaber/Cerebro/blob/master/Cerebro/saved-models/emotions_model.f5) then add them to new foldercalled ```saved-models```
***Using Landmark*** : if you want to use Landmark feature extractor you have to dowenload this file landmarks with 68 point using this [link](https://github.com/AmrSaber/Cerebro/tree/master/Cerebro/saved-models/face-landmarks) in this path
```saved-models/face-landmarks``` "create new folder called landmarks in saved-models" 

**Video demo**

**Real Time demo**

**Maintainers** 

* [AmrSaber](https://github.com/AmrSaber)
* [Wafaaismail](https://github.com/Wafaaismail)
* [aim97](https://github.com/aim97)
* [ManarArabi](https://github.com/ManarArabi)
* [SalmaSayed](https://github.com/SalmaSayed)
* [MuhammeaAef](https://github.com/MuhammadAref)


