from model import predict
from image import *
from reader import emotions_map

def extract_faces_emotion(image):

    item = []
    items= []

    #each item has [face_image, corner_coordinate, emotion]
    #faces of size 48*48
    #corners coordinates as ()

    emotions_count = len(set(emotions_map))
    model = EmotionsModel(emotions_count, use_hog=true)
    for i in range (faces) :
        item.append(faces[i])
        item.append(corners_coordinate[i])
        emotion = model.predict(faces[i])
        item.append(emotion)
        items.append(item)
        item.clear
    return items

def mark_faces_emotions(image):
    #not implemented yet
    return new_image
