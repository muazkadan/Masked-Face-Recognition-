import os
import cv2
import random as rnd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import numpy as np

face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

IMG_SIZE = 274
DataDir = 'Datasets/training_dataset'
CATEGORIES = ["Masked", "Unmasked"]
training_data = []

for category in CATEGORIES:
    path = os.path.join(DataDir, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        image_faces = face_cascade.detectMultiScale(img_array, 1.3, 5)
        if(len(image_faces) == 0):
            os.remove(path + "/" + img.title())
        for face in image_faces:
            x, y, w, h = [v for v in face]
            face_crop = img_array[y:y + h, x:x + w]
            new_array = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])

print("Number of samples: " + str(len(training_data)))

rnd.shuffle(training_data)

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

X = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)
# y = to_categorical(y, len(CATEGORIES))

model = Sequential()
model.add(Conv2D(64, (3, 3), activation="relu", input_shape=X.shape[1:]))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X, y, epochs=10, batch_size=25, validation_split=0.1)

savedModelPath = "SavedModel"
model.save(savedModelPath)

# cv2.imshow('face', training_data[16][0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
