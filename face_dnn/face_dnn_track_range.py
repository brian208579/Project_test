import numpy as np
import cv2
import time

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

KNOWN_DISTANCE = 60 
KNOWN_WIDTH_M = 14.5
KNOWN_WIDTH_F = 13.3
KNOWN_WIDTH = 13.9

net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")
gender_list = ['Male', 'Female'] 
MODEL_MEAN_VALUES = (78.43, 87.77, 114.9)

# compute the FocalLengh
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
            
# setting the Camera
cam = cv2.VideoCapture(0)  # 如果筆電要外裝 Camera 要把 0 改成 1
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(0.1)
direction = ""
(dirX, dirY) = ("", "")

while True:
    x_y = []
    list_h = []
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
            
            if startX <= 0 or startY <= 0 or endX >= 640 or endY >= 480 :
                continue
            
            x_y = ((endY-startY), (startX, startY, endX, endY))
            # print(x_y)
            list_h.append(x_y)
            (_, target) = max(list_h)
            (targ_sx, targ_sy, targ_ex, targ_ey) = target
            (centerX, centerY) = (int((targ_sx + targ_ex)/2), int((targ_sy + targ_ey)/2))
            
            # (centerX, centerY) = (int((startX+endX)/2), int((startY+endY)/2)) 
            
            if centerX > 320 + 50:
                dirX = "MoveRignt"
            if centerX < 320 - 50:
                dirX = "MoveLeft"
            if centerX < 320 + 50 and centerX > 320 -50:
                dirX = ""
            if centerY > 240 + 30:
                dirY = "Movedown"
            if centerY < 240 - 30:
                dirY = "MoveUp"
            if centerY < 240 + 30 and centerY > 240 - 30:
                dirY = ""
            if dirX != "" and dirY != "": 
                direction = "{}-{}".format(dirY, dirX)
            else:
                if dirX != "":
                    direction = dirX 
                else:
                    direction = dirY
            cv2.putText(img, direction, (img.shape[1] - 630, img.shape[0] - 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            (x1, x2, y1, y2) = rectangle_resize(startX, endX, startY, endY)
            face_img = img[y1:y2, x1:x2].copy()
            blob_2 = cv2.dnn.blobFromImage(face_img, 5, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob_2)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            gender_proba = "{:.1f}%".format(gender_preds.max() * 100)
            if gender_preds.max() > 0.8 :
                if gender == 'Male':
                    cm = distance_to_camera(KNOWN_WIDTH_M, focalLength, (endX-startX))
                else:
                    cm = distance_to_camera(KNOWN_WIDTH_F, focalLength, (endX-startX))
            else :
                gender = "Uncertain"
                cm = distance_to_camera(KNOWN_WIDTH, focalLength, (endX-startX))
                
            # cv2.putText(img, gender_proba, (startX, startY+(endY-startY)+65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
            # cv2.putText(img, "%s" %(gender), (startX, (startY-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 100, 0), 2)
            # cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(img, "%dcm" %(cm), (startX, startY+(endY-startY)+30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
            cv2.rectangle(img, (270, 210), (370, 270), (255, 30, 0), 3)
            cv2.circle(img, (centerX, centerY), 8, (0, 0, 255), -1, 10)
            cv2.putText(img, "CenterX=%d" %(centerX), (img.shape[1] - 630, img.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 240, 0), 2)
            cv2.putText(img, "CenterY=%d" %(centerY), (img.shape[1] - 630, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 240, 0), 2)
            cv2.line(img, (centerX, 0), (centerX, 480), (0, 255, 0), 2) 
            cv2.line(img, (0, centerY), (640, centerY), (0, 255, 0), 2) 
    cv2.putText(img, "Found %d face(s)" % (face_len), (img.shape[1] - 270, img.shape[0] - 445), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 2)
    # cv2.putText(img, "x1={},y1={},x2={},y2={}".format(x1,y1,x2,y2), (img.shape[1] - 630, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 215, 255), 2)
    cv2.putText(img, "%dFPS" % (frame_rate), (img.shape[1] - 160, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 215, 255), 2)
    cv2.imshow('img', img)
    t2 = cv2.getTickCount()
    frame_rate = FPS_calculate(t1 ,t2)
    
    if cv2.waitKey(1) & 0xFF == 27: # press esc exit
        break

cam.release()
cv2.destroyAllWindows()


