import numpy as np
import cv2
import time
from collections import deque

# Calculate fps
frame_rate = 1
freq = cv2.getTickFrequency()
def FPS_calculate(T1, T2):
    time1 = (T2-T1)/freq
    frame_rate = 1/time1
    return(frame_rate)

def rectangle_resize(X1, X2, Y1, Y2):
      x1_res = int(X1*0.9)
      x2_res = int(X2*1.1)
      y1_res = int(Y1*0.8)
      y2_res = int(Y2*1)
     
      if x1_res <= 0:
          x1_res = 0
      else:
          x1_res = x1_res
      if y1_res <= 0:
          y1_res = 0
      else:
          y1_res = y1_res
      if x2_res >= img.shape[1]:
          x2_res = img.shape[1]
      else:
          x2_res = x2_res
      if y2_res >= img.shape[0]:
          y2_res = img.shape[0]
      else:
          y2_res = y2_res
          return(x1_res, x2_res, y1_res, y2_res)
    

(KNOWN_DISTANCE, KNOWN_WIDTH, KNOWN_WIDTH_M, KNOWN_WIDTH_F) = (60, 13.9, 14.5, 13.3)

face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")
gender_list = ['Male', 'Female'] 

# Calculate FocalLengh
img = cv2.imread("test1.jpg")
(h, w) = img.shape[:2]
img_res = cv2.resize(img, (300, 300))
blob_1 = cv2.dnn.blobFromImage(img_res, 2, (300, 300), (104, 177, 123), swapRB=False)
face_net.setInput(blob_1)
detections = face_net.forward()
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence < 0.9:
        continue
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (X1, Y1, X2, Y2) = box.astype("int")
    focalLength = ((X2-X1) * KNOWN_DISTANCE) / KNOWN_WIDTH_M  # calculate the focalLengh ( w = pixel width )
    break
            
# setting the Camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# cam.set(cv2.CAP_PROP_FPS, 10)
time.sleep(0.1)

pts = deque(maxlen = 30) 
direction = ""
velocity = ""
(dX, dY) = (0, 0)
counter = 0
while True:
    face_len = 0
    t1 = cv2.getTickCount()
    ret, img = cam.read()
    (h, w) = img.shape[:2]
    # Face detection
    img_res = cv2.resize(img, (300, 300))
    blob_1 = cv2.dnn.blobFromImage(img_res, 2, (300, 300), (104, 177, 123), swapRB=False)
    face_net.setInput(blob_1)
    detections = face_net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.9:
            face_len += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int") 
            face_proba = "{:.1f}%".format(confidence * 100)
            
            if startX <= 0 or startY <= 0 or endX >= 640 or endY >= 480 :
                continue
            
            # Tracking & Driection
            (centerX, centerY) = (int((startX+endX)/2), int((startY+endY)/2)) 
            center = (centerX, centerY)
            pts.appendleft(center)
            for j in range(1, len(pts)):
                if pts[j-1] is None or pts[j] is None:
                       continue
                thickness = int(np.sqrt(60 / float(j + 1)) * 2.5)
                cv2.line(img, pts[j-1], pts[j], (255, 0, 255), thickness)    
                
                if counter >= 10 and j == 1 and pts[-10] is not None: # j = 1 是為了不要讓方向 print 太快
                    dX = pts[-10][0] - pts[j][0]
                    dY = pts[-10][1] - pts[j][1]
                    (dirX, dirY) = ("", "")
                    
                    if (np.abs(dX) > 30 and np.abs(dX) < 50) or (np.abs(dY) > 30 and np.abs(dY) < 50):
                        velocity = "slow"
                    elif (np.abs(dX) > 50 and np.abs(dX) < 100) or (np.abs(dY) > 50 and np.abs(dY) < 100):
                        velocity = "medium"
                    elif (np.abs(dX) > 100 ) or (np.abs(dY) > 100):
                        velocity = "fast"
                    else:
                        velocity = ""
                    
                    if np.abs(dX) > 30:
                        if np.sign(dX) == 1: 
                            dirX = "East"
                        else: 
                            dirX = "West"
                    if np.abs(dY) > 30:
                        if np.sign(dY) == 1: 
                            dirY = "North"
                        else: 
                            dirY = "South"
                    if dirX != "" and dirY != "": 
                        direction = "{}-{}".format(dirY, dirX)
                    else:
                        if dirX != "":
                            direction = dirX 
                        else:
                            direction = dirY 
                    cv2.putText(img, direction, (img.shape[1] - 620, img.shape[0] - 440), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    cv2.putText(img, velocity, (img.shape[1] - 620, img.shape[0] - 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    
            # Gender detection
            (x1, x2, y1, y2) = rectangle_resize(startX, endX, startY, endY)       
            face_img = img[y1:y2, x1:x2]
            blob_2 = cv2.dnn.blobFromImage(face_img, 5, (227, 227), (78.43, 87.77, 114.90), swapRB=False)
            gender_net.setInput(blob_2)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            gender_proba = "{:.1f}%".format(gender_preds.max() * 100)
            if gender_preds.max() > 0.8 :
                if gender == 'Male':
                    cm = (KNOWN_WIDTH_M*focalLength)/(endX-startX)
                else:
                    cm = (KNOWN_WIDTH_F*focalLength)/(endX-startX)
            else :
                gender = "Uncertain"
                cm = (KNOWN_WIDTH*focalLength)/(endX-startX)
                
            cv2.putText(img, gender_proba, (startX, startY+(endY-startY)+65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
            cv2.putText(img, "%s" %(gender), (startX, (startY-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 100, 0), 2)
            cv2.putText(img, "%dcm" %(cm), (startX, startY+(endY-startY)+30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)        
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.circle(img, (centerX, centerY), 7, (0, 0, 255), -1, 10)
            cv2.putText(img, "CenterX=%d" %(centerX), (img.shape[1] - 630, img.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 240, 0), 2)
            cv2.putText(img, "CenterY=%d" %(centerY), (img.shape[1] - 630, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 240, 0), 2)
    cv2.putText(img, "Found %d face(s)" % (face_len), (img.shape[1] - 270, img.shape[0] - 445), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 2)
    cv2.putText(img, "%dFPS" % (frame_rate), (img.shape[1] - 160, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 215, 255), 2)
    # cv2.putText(img, "x1={},y1={},x2={},y2={}".format(x1,y1,x2,y2), (img.shape[1] - 630, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 215, 255), 2)
    cv2.imshow('img', img)
    t2 = cv2.getTickCount()
    frame_rate = FPS_calculate(t1 ,t2)
    counter += 1
    if cv2.waitKey(1) & 0xFF == 27: # press esc exit
        break
cam.release()
cv2.destroyAllWindows()

# x1_res = int(startX*0.9)
# x2_res = int(endX*1.1)
# y1_res = int(startY*1)
# y2_res = int(endY*1)
# if x1_res <= 0:
#     x1_res = 0
# else:
#     x1_res = x1_res
# if y1_res <= 0:
#     y1_res = 0
# else:
#     y1_res = y1_res
# if x2_res >= img.shape[1]:
#     x2_res = img.shape[1]
# else:
#     x2_res = x2_res
# if y2_res >= img.shape[0]:
#     y2_res = img.shape[0]
# else:
#     y2_res = y2_res
# (x1, x2, y1, y2) = (x1_res, x2_res, y1_res, y2_res)
# print("startX={},startY={},endX={},endY={}".format(startX, startY, endX, endY))
# print("x1={},y1={},x2={},y2={}".format(x1, y1, x2, y2))