import cv2
import numpy as np
import os

haar_file = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

datasets = 'E:\Dhanush_Main_Projects\Face Recognition using opencv\datasets'

print('Training....')

(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subject_path = os.path.join(datasets, subdir)
        for filename in os.listdir(subject_path):
            path = os.path.join(subject_path, filename)
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
            print(labels)
            
(width, height) = (130, 100)
(images, labels) = [np.array(lis) for lis in [images, labels]]

# Train the dataset
#model = cv2.face.FisherFaceRecognizer_create()
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Testing the datasets
face_cascade = cv2.CascadeClassifier(haar_file)

# Opening webcam
web_cam = cv2.VideoCapture(0)
cnt = 0

while True:
    (_, im) = web_cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detecting the face through the webcam
    
    # Creating the rectangle bounding box
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if prediction[1] < 800:
            cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (51, 255, 255))
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt = 1
            cv2.putText(im, 'unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if cnt > 100:
                print("unknown")
                cv2.imwrite("input.jpg", im)
                cnt = 0
                
        cv2.imshow('opencv', im)
        key = cv2.waitKey(10)
        if key == ord("a"):
            break

web_cam.release()
cv2.destroyAllWindows()
