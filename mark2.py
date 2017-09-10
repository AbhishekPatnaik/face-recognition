import cv2
import numpy as np
import matplotlib.pyplot as plt

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_detect = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
cam = cv2.VideoCapture(0)
sample = 0
id='abhishek'
while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        sample = sample + 1
        cv2.imwrite('/home/abhishek/testdatasets/'+str(id)+'.'+str(sample)+'.jpeg', gray[x:x+w, y:y+h])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        plt.imshow(img)
        plt.show()
        if sample == 50:
            break

cam.release()
cv2.destroyAllWindows()


