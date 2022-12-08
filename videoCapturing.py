import cv2
import numpy as np
import os
img = cv2.VideoCapture(0)
path = "imports\\haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(path)
skip = 0
face_data = []
labels = []
names = {}
class_id = 0
face_section = np.zeros((100, 100), dtype='uint8')
dirpath = "imgs"
for file in os.listdir("imgs"):
    if file.endswith(".npy"):
        data_item = np.load(dirpath+'\\'+file)
        names[class_id] = file[:-4]
        face_data.append(data_item)
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))


def dist(X1, X2):
    return np.sqrt(np.sum((X1-X2)**2))


def knn(X, Y, Query, k=5):
    m = X.shape[0]
    vals = []
    for i in range(m):
        d = dist(Query, X[i])
        vals.append((d, Y[i]))
    vals = sorted(vals, key=lambda x: x[0])[:k]
    vals = np.array(vals)
    new_vals = np.unique(vals[:, 1], return_counts=True)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    return pred


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
        out = knn(face_dataset, face_labels, face_section.flatten())
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Camera", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('s'):
        break
img.release()
cv2.destroyAllWindows()
