from model import predict
from reader import emotions_map

def extract_faces_emotion(image, detector_type):
    """
    detector_type >> ('dlib, haar, lbp')
    each item has [face_image, corner_coordinate, emotion]
    faces of size 48*48 , gray scale
    corners coordinates as ()
    """
    item = []
    items= []

    if detector_type == 'dlib':
        from image.face_detector import dlib as detector
    elif detector_type =='haar':
        from image.face_detector import haar as detector
    elif detector_type =='lbp':
        from image.face_detector import lbp as detector
    else :
        raise Exception("invalid detector")

    faces = detector.get_faces(image)
    emotions_count = len(set(emotions_map))
    model = EmotionsModel(emotions_count, use_hog=true)
    for i in range (faces) :
        item.append(faces[i][0]) #face
        item.append(faces[i][1]) #corner coordinates
        emotion = model.predict(faces[i][0])
        item.append(emotion)
        items.append(item)
        item.clear
    return items

def mark_faces_emotions(image):
    #not implemented yet
    return new_image

def select_detector_type(detector_type):
