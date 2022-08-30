import numpy as np
import cv2

def resize(img, new_width=700):
    height = img.shape[0]
    width = img.shape[1]
    ratio = height/width
    new_height = int(ratio*new_width)
    return cv2.resize(img,(new_width,new_height))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#cap = cv2.VideoCapture('person_walking.mp4')
cap = cv2.VideoCapture(0)
while True:
    _,frame = cap.read()
    #frame = cv2.imread('2.jpg')
    frame = resize(frame)
    detections = face_cascade.detectMultiScale(frame, scaleFactor= 1.05, minNeighbors=6)

    for face in detections:
        x,y,w,h = face
        #frame[y:y+h,x:x+w] = cv2.GaussianBlur(frame[y:y+h,x:x+w], (201,201), cv2.BORDER_DEFAULT)
        frame[y:y + h, x:x + w] = cv2.blur(frame[y:y + h, x:x + w], (25, 25))
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) == 30:
            break

#cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()