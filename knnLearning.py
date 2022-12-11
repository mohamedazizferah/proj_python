import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

img = cv2.VideoCapture(0)
path = "imports\\haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(path)
skip = 0
face_data = []
labels = []
names = {}
class_id = 0
n_neighbors = 0
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
    n_neighbors = n_neighbors + 1
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
vector = np.vectorize(np.int_)
pred_name = np.concatenate(face_labels, axis=0)
a = vector(pred_name)
range = 0
y = []
iter = len(a)

for range in a:
    y.append(names[range])
    range += 1

X_train, X_test, y_train, y_test = train_test_split(
    face_dataset, y, train_size=0.99, stratify=face_labels)

# knn = KNeighborsClassifier(n_neighbors=250, weights='uniform', algorithm='auto',
#                            leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
