import tensorflow as tf


model = tf.keras.models.load_model("BestSavedModel")

savedModelPath = "BestSavedModel"

converter = tf.lite.TFLiteConverter.from_saved_model(savedModelPath)
tflite_model = converter.convert()
open("TFLite/ConvertedModel.tflite", "wb").write(tflite_model)
