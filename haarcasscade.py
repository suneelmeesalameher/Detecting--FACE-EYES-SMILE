import numpy as np
import time
import cv2

a=0

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

smile_cascade=cv2.CascadeClassifier('haarcascade_smile.xml')

video = cv2.VideoCapture(0)

while 1:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
        smile = smile_cascade.detectMultiScale(roi_gray)
        for(sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
            
    cv2.imshow('Capturing',frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
