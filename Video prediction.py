from tensorflow import keras
import cv2
import numpy as np
from playsound import playsound
import time
from threading import Thread, active_count


def warn():
    playsound('Sound Effects/uyari.mp3')
    time.sleep(4)


model = keras.models.load_model("BestSavedModel")

IMG_SIZE = 274

face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('Cascades/test_cascade.xml')

# eye_cascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
# LBP_cascade = cv2.CascadeClassifier('Cascades/lbpcascade_profileface.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
print(active_count())
i = 0
while True:
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    detectedFrame = frame
    masked = 0
    unmasked = 0
    for face in faces:
        x, y, w, h = [v for v in face]
        face_crop = frame[y:y + h, x:x + w]
        test_array = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
        test_array = np.array(test_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        img_class = model.predict(test_array)
        if (img_class > 0.5).astype("int32")[0][0] == 0:
            detectedFrame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            masked += 1
        else:
            detectedFrame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            unmasked += 1
            if active_count() <= 1:
                Thread(target=warn).start()
            # print(active_count())
    cv2.putText(frame, "Masked: " + str(masked) + " Unmasked: " + str(unmasked),
                (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)
    cv2.imshow('frame', detectedFrame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
