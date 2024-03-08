import cv2
import os

#haar_file = r'haarcascade_frontalface_default.xml'

# Get the path to the pre-trained face cascade
haar_file = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

# Rest of your code...
datasets = 'E:\Dhanush_Main_Projects\Face Recognition using opencv\datasets'
sub_data = 'New'
path = os.path.join(datasets, sub_data)

# Create the 'datasets' directory if it doesn't exist
if not os.path.exists(datasets):
    os.makedirs(datasets)

# Create the subdirectory 'New' inside 'datasets'
if not os.path.exists(path):
    os.mkdir(path)

(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file)
web_cam = cv2.VideoCapture(0)
count = 1

while count < 51:
    print(count)
    (_, im) = web_cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite(os.path.join(path, '%s.png' % count), face_resize)
    
    count += 1
    
    cv2.imshow('opencv', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

web_cam.release()
cv2.destroyAllWindows()
