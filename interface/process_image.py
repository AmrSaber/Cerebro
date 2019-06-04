import sys;
sys.path.insert(1, '../model')
sys.path.insert(1, '../image')
sys.path.insert(1, '../saved-models/emotions_model.f5')
sys.path.insert(1,'../image/face_detector')
import model
import cv2

def extract_faces_emotions(image, detector_type = 'dlib'):

    items= []
    if detector_type == 'dlib':
        from face_detector import detect_dlib as detector
    elif detector_type =='haar':
        from face_detector import detect_haar as detector
    elif detector_type =='lbp':
        from face_detector import detect_lbp as detector
    else :
        raise Exception("invalid detector")

    faces = detector.get_faces(image)
    print(len(faces))
    if len(faces):

        print(faces[0][0])
        
    # emotions_count = len(set(emotions_map))
    m = model.EmotionsModel(7 , use_hog=True ,use_lm =True ,use_cnn =False)
    for i in range (len(faces)) :
        item = []
        item.append(faces[i][0]) #face
        item.append(faces[i][1]) #corner coordinates
        emotion = m.predict(faces[i][0])
        item.append(emotion)
        items.append(item)
    return items

def mark_faces_emotions(image, detector_type = 'dlib', extracted_faces_emotions = []):
    """
    if detector_type >> None : don't detect
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 51, 51)
    offest_x = 2
    offest_y = 5
    if detector_type != "None":
        extracted_faces_emotions = extract_faces_emotions(image, detector_type)
    """
    extracted_faces_emotions[i] >>
                item (face, corners(topright,bottomleft), emotion)
    extracted_faces_emotions[i][1] >> corners
    extracted_faces_emotions[i][1][0] >> topright
    extracted_faces_emotions[i][1][0][0] >> toprightX
    extracted_faces_emotions[i][1][0][1] >> toprightY
    extracted_faces_emotions[i][2] >> emotion

    """
    if extracted_faces_emotions != None :
        for i in range(len(extracted_faces_emotions)):
            tmp = (extracted_faces_emotions[i][1][0][0]-offest_x,
                   extracted_faces_emotions[i][1][0][1]-offest_y)

            image = cv2.rectangle(image,
                                  extracted_faces_emotions[i][1][0],
                                  extracted_faces_emotions[i][1][1],
                                  (66,206,244),
                                  2)

            image = cv2.putText(image,
                                extracted_faces_emotions[i][2],
                                tmp,
                                font,
                                font_scale,
                                font_color,
                                2)

    return image
