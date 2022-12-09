import cv2
import numpy as np

img = cv2.VideoCapture(0)
path = "imports\\haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(path)
face_data = []
skip = 0
face_section = np.zeros((100, 100), dtype='uint8')
dirpath = "imgs\\"

name = input("Enter your name: ")
while True:
    ret, frame = img.read()
    if ret == False:
        continue
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        face_section = gray_frame[y:y + h, x:x + w]
        face_section = cv2.resize(face_section, (100, 100))
        cv2.putText(frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        face_data.append(face_section)
        print(len(face_data))
    cv2.imshow("Camera", frame)
    if skip % 5 == 0:
        face_data.append(face_section)
    skip += 1
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('s'):
        break
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
np.save(dirpath + name + '.npy', face_data)
img.release()
cv2.destroyAllWindows()
