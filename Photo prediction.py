from tensorflow import keras
import cv2
import numpy as np

IMG_SIZE = 274
CATEGORIES = ["Masked", "Unmasked"]

face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')


model = keras.models.load_model("SavedModel")

test_img = cv2.imread('Datasets/test_data/0003.jpg', cv2.IMREAD_COLOR)
image_faces = face_cascade.detectMultiScale(test_img, 1.3, 5)
print("number of faces " + str(len(image_faces)))
if len(image_faces) != 0:
    for face in image_faces:
        x, y, w, h = [v for v in face]
        face_crop = test_img[y:y + h, x:x + w]
        test_array = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
        cv2.imshow('face', test_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    test_array = np.array(test_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    img_class = model.predict([test_array])
    print((img_class > 0.5).astype("int32"))
    # print(CATEGORIES[int(img_class[0][0])])