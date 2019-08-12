import numpy as np
import cv2
import dlib
from matplotlib import pyplot as plt

KNOWN_DISTANCE = 60 
KNOWN_WIDTH_M = 14.5
KNOWN_WIDTH_F = 13.3

net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")
gender_list = ['Male', 'Female'] 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# face detection
img = cv2.imread("test1.jpg")
(h, w) = img.shape[:2]
img_res = cv2.resize(img, (300, 300))
blob_1 = cv2.dnn.blobFromImage(img_res, 1, (300, 300), (104, 177, 123))
net.setInput(blob_1)
detections = net.forward()
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.8:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (X1, Y1, X2, Y2) = box.astype("int")
        focalLength = ((X2-X1) * KNOWN_DISTANCE) / KNOWN_WIDTH_M  # calculate the focalLengh ( w = pixel width )
        break

def distance_to_camera(knownWidth, focalLength, perWidth): # calculate the distance ( cm ) 
    return ((knownWidth * focalLength) / perWidth)

landmark_predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

face_len = 0
img = cv2.imread("face3.jpeg")
(h, w) = img.shape[:2]
img_res = cv2.resize(img, (300, 300))
blob_1 = cv2.dnn.blobFromImage(img_res, 1, (300, 300), (104, 177, 123))
net.setInput(blob_1)
detections = net.forward()
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.8:
        face_len = face_len + 1
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        proba = confidence * 100  # proba = "{:.1f}%".format(confidence * 100)
        
        # x1 = int(startX*0.9)
        # x2 = int(endX*1.1)
        # y1 = int(startY*1)
        # y2 = int(endY*1)
        
        x1 = startX
        x2 = endX
        y1 = startY
        y2 = endY
        
        face_img = img[y1:y2, x1:x2].copy()
        # plt.axis("off")
        # plt.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        # plt.show()
        blob_2 = cv2.dnn.blobFromImage(face_img, 3, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob_2)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        gender_proba = gender_preds[0].max()*100
        # print(gender_preds[0])
        # print(gender_preds[0].argmax())
        if gender == 'Male':
            cm = distance_to_camera(KNOWN_WIDTH_M, focalLength, (endX-startX))
        else:
            cm = distance_to_camera(KNOWN_WIDTH_F, focalLength, (endX-startX))
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)    
        # cv2.putText(img, "%s" % (gender), (startX, (startY-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 100, 0), 2)
        cv2.putText(img, "%.2f" % ((endX-startX)), (startX, startY+(endY-startY)+30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
        # cv2.putText(img, "%.2f" % (proba), (startX, startY+(endY-startY)+65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
        # cv2.putText(img, "%dcm" %(cm), (startX, startY+(endY-startY)+30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)        
cv2.putText(img, "Found %d face(s)" % (face_len), (img.shape[1] - 340, img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 140, 255), 2)
cv2.imshow('img', img)
    
cv2.waitKey(0)
cv2.destroyAllWindows()


