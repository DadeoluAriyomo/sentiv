import tensorflow as tf

keras_model_path = "best_fer_model.keras"
tflite_model_path = "best_fer_model.tflite"

model = tf.keras.models.load_model(keras_model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Converted {keras_model_path} to {tflite_model_path}")
