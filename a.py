import cv2
import numpy as np
import os
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    "imports\\haarcascade_frontalface_alt.xml")
skip = 0
face_data = []
labels = []
names = {}
class_id = 0
face_section = np.zeros((100, 100), dtype="uint8")
dirpath = "imgs"
# name = input("Enter your name")
for file in os.listdir("imgs"):
    if file.endswith(".npy"):
        data_item = np.load(dirpath+'\\'+file)
        print(file)
        print("dataitem", data_item)
        names[class_id] = file[:-4]
        face_data.append(data_item)
        print("face_data", face_data)
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)
face_dataset = np.concatenate(face_data, axis=0)
print(f"facedataset {face_dataset} len= {len(face_dataset)}")
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))


def dist(X1, X2):
    return np.sqrt(np.sum((X1-X2)**2))


def knn(X, Y, Query, k=5):
    m = X.shape[0]
#     print(Query.shape)
    vals = []
    for i in range(m):
        #         print(Query[i].shape,X[i].shape)
        d = dist(Query, X[i])
        vals.append((d, Y[i]))

    vals = sorted(vals, key=lambda x: x[0])[:k]
    vals = np.array(vals)

    new_vals = np.unique(vals[:, 1], return_counts=True)
#     print(new_vals)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    return pred


while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2]*f[3])
    for face in faces[-1:]:
        x, y, w, h = face

        offset = 10
        face_section = gray_frame[y-offset:y+h+offset, x-offset:w+x+offset]
        face_section = cv2.resize(face_section, (100, 100))
        pred = knn(face_dataset, face_labels, face_section.flatten())
        pred_name = names[int(pred)]
        cv2.putText(frame, pred_name, (x, y-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10)

    cv2.imshow("camera", frame)
#     if skip%10 == 0:
#         face_data.append(face_section)
#     skip += 1
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
# face_data = np.asarray(face_data)
# face_data = face_data.reshape((face_data.shape[0],-1))
# np.save(dirpath+'\\'+name+'.npy',face_data)
# print(face_data.shape)
cap.release()
cv2.destroyAllWindows()
