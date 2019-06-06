from model import emotions_model as model
import cv2

def extract_faces_emotions(image, detector_type = 'dlib'):

    items= []
    if detector_type == 'dlib':
        from image.face_detector import detect_dlib as detector
    elif detector_type =='haar':
        from image.face_detector import detect_haar as detector
    elif detector_type =='lbp':
        from image.face_detector import detect_lbp as detector
    else :
        raise Exception("invalid detector")

    faces = detector.get_faces(image)

    # emotions_count = len(set(emotions_map))
    emotionsModel = model.EmotionsModel()
    for i in range (len(faces)) :
        item = []
        item.append(faces[i][0]) #face
        item.append(faces[i][1]) #corner coordinates
        if faces[i][0].shape[0] == 0 or faces[i][0].shape[1] == 0: continue
        emotion = emotionsModel.predict(faces[i][0])
        item.append(emotion)
        items.append(item)
    return items

def mark_faces_emotions(image, detector_type = 'dlib', extracted_faces_emotions = None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (72, 1, 68)
    offset_x = 5
    offset_y = 0

    # if detector_type >> None : don't detect
    if detector_type != None and extracted_faces_emotions == None:
        extracted_faces_emotions = extract_faces_emotions(image, detector_type)

    """
    extracted_faces_emotions[i] >> item (face, corners(topright,bottomleft), emotion)
    extracted_faces_emotions[i][1] >> corners
    extracted_faces_emotions[i][1][0] >> topright
    extracted_faces_emotions[i][1][0][0] >> toprightX
    extracted_faces_emotions[i][1][0][1] >> toprightY
    extracted_faces_emotions[i][2] >> emotion
    """

    if extracted_faces_emotions == None: return image

    for item in extracted_faces_emotions:
        tmp = (item[1][0][0] + offset_x, item[1][0][1] - offset_y)

        #text background
        size = cv2.getTextSize(item[2], font, fontScale=font_scale, thickness=1)[0]
        box_coords = (tmp, (tmp[0] + size[0] - 2, tmp[1] - size[1] - 2))
        cv2.rectangle(image, box_coords[0], box_coords[1], color, cv2.FILLED)

        #selected face box
        image = cv2.rectangle(
            image,
            item[1][0],
            item[1][1],
            color,
            2,
        )

        #text
        image = cv2.putText(
            image,
            item[2],
            tmp,
            font,
            font_scale,
            (255, 255, 255),
            1,
        )

    return image
