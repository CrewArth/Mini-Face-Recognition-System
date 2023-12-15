import cv2
import numpy as np 
import os

cascadeClassifer = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

people = []
for i in os.listdir(r'E:\OpenCV\Faces'):
    people.append(i)

# print(people)
    
dir = r'E:\OpenCV\Faces'

feature = []
labels = []

def createTrain():
    for person in people:
        path = os.path.join(dir, person)
        label = people.index(person)
        
        for image in os.listdir(path):
            image_path = os.path.join(path,image)

            image_array = cv2.imread(image_path)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            face_rect = cascadeClassifer.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in face_rect:
                face_region = gray[y:y+h, x:x+w]
                feature.append(face_region)
                labels.append(label)

createTrain()
print('Training of Model is Completed....')
# print(f'The Length of Features is = {len(feature)}')
# print(f'The Length of Labels is = {len(labels)}')

feature = np.array(feature, dtype='object')
labels = np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train Recognizer on Feature and Label List.
face_recognizer.train(feature, labels)

face_recognizer.save('facesTrained.yml')
np.save('feature.npy',feature)
np.save('labels.npy ',labels)
