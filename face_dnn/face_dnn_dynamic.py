import numpy as np
import cv2
import time

KNOWN_DISTANCE = 60 
KNOWN_WIDTH_M = 14.5
KNOWN_WIDTH_F = 13.3

net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")
gender_list = ['Male', 'Female'] 
MODEL_MEAN_VALUES = (78.43, 87.77, 114.90)

# face detection
img = cv2.imread("test1.jpg")
(h, w) = img.shape[:2]
img_res = cv2.resize(img, (300, 300))
blob_1 = cv2.dnn.blobFromImage(img_res, 1.0, (300, 300), (104, 177, 123))
net.setInput(blob_1)
detections = net.forward()
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence < 0.9:
        continue
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (X1, Y1, X2, Y2) = box.astype("int")
    focalLength = ((X2-X1) * KNOWN_DISTANCE) / KNOWN_WIDTH_M  # calculate the focalLengh ( w = pixel width )
    break

# Calculate distance
def distance_to_camera(knownWidth, focalLength, perWidth): # calculate the distance ( cm ) 
    distance = (knownWidth * focalLength) / perWidth
    return (distance)

# Calculate fps
frame_rate = 1
freq = cv2.getTickFrequency()
def FPS_calculate(T1, T2):
    time1 = (T2-T1)/freq
    frame_rate = 1/time1
    return(frame_rate)
    
def rectangle_resize(X1, X2, Y1, Y2):
     x1 = int(X1*0.9)
     x2 = int(X2*1.1)
     y1 = int(Y1*1)
     y2 = int(Y2*1)
     return(x1, x2, y1,y2)
     
def over_size(x1, x2, y1, y2):
    if x1 < 0:
        x1 = 0
    else:
        x1 = x1
    if y1 < 0:
        y1 = 0
    else:
        y1 = y1
    if x2 > img.shape[1]:
        x2 = img.shape[1]
    else:
        x2 = x2
    if y2 > img.shape[0]:
        y2 = img.shape[0]
    else:
        y2 = y2
        return(x1, x2, y1, y2)
 
# setting the Camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(0.1)

while True:
    face_len = 0
    t1 = cv2.getTickCount()
    ret, img = cam.read()
    (h, w) = img.shape[:2]
    img_res = cv2.resize(img, (300, 300))
    blob_1 = cv2.dnn.blobFromImage(img_res, 2, (300, 300), (104, 177, 123))
    net.setInput(blob_1)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.9:
            face_len += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int") 
            proba = "{:.1f}%".format(confidence * 100)
            
            (x1, x2, y1, y2) = rectangle_resize(startX, endX, startY, endY)
            (x1, x2, y1, y2) = over_size(x1, x2, y1, y2)
            # print("x1={},y1={},x2={},y2={},face_len={}".format(x1,y1,x2,y2,face_len))
            
            face_img = img[y1:y2, x1:x2].copy()
            blob_2 = cv2.dnn.blobFromImage(face_img, 5, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob_2)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            if gender == 'Male':
                cm = distance_to_camera(KNOWN_WIDTH_M, focalLength, (endX-startX))
            else:
                cm = distance_to_camera(KNOWN_WIDTH_F, focalLength, (endX-startX))
            cv2.putText(img, proba, (startX, startY+(endY-startY)+65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
            cv2.putText(img, "%s" % (gender), (startX, (startY-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 100, 0), 2)
            cv2.putText(img, "%dcm" %(cm), (startX, startY+(endY-startY)+30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)        
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(img, "Found %d face(s)" % (face_len), (img.shape[1] - 270, img.shape[0] - 445), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 2)
    cv2.putText(img, "%dFPS" % (frame_rate), (img.shape[1] - 160, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 215, 255), 2)
    # cv2.putText(img, "x1={},y1={},x2={},y2={}".format(x1,y1,x2,y2), (img.shape[1] - 630, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 2)
    cv2.imshow('img', img)
    t2 = cv2.getTickCount()
    frame_rate = FPS_calculate(t1 ,t2)
    # cv2.putText(img, "Not have face", (img.shape[1] - 350, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 215, 255), 2)
    
    if cv2.waitKey(1) & 0xFF == 27: # press esc exit
        break

cam.release()
cv2.destroyAllWindows()