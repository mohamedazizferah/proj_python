from knnLearning import knn
import cv2
import numpy as np
import os

img = cv2.VideoCapture(0)
path = "imports\\haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(path)
face_section = np.zeros((100, 100), dtype='uint8')
dirpath = "imgs"


def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1


while True:
    ret, frame = img.read()
    if ret == False:
        continue
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        # face_section = gray_frame[y:y + h, x:x + w]
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))
        out = knn.predict(face_section.flatten().reshape(1, -1))
        a = listToString(out)
        cv2.putText(frame, a, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Camera", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('s'):
        break
img.release()
cv2.destroyAllWindows()
