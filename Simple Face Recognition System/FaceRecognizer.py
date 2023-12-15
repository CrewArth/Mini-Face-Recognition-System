import numpy as np 
import cv2
import os

cascadeClassifer = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

people = []
for i in os.listdir(r'E:\OpenCV\Faces'):
    people.append(i)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('facesTrained.yml')

# Reading the Image
img = cv2.imread(r'E:\OpenCV\Testing\test8.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Person', gray)
cv2.resize(img, dsize= (100,80))
face_rect = cascadeClassifer.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
for (x,y,w,h) in face_rect:
    face_region = gray[y:y+h, x:x+w]
    
    label, confidence = face_recognizer.predict(face_region)
    print(f'Label : {label}')
    print(f'Confidence Rate : {confidence}')
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
    cv2.putText(img, str(people[label]), (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0))

cv2.imshow('Detected Face', img)
cv2.waitKey(0)
