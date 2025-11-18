# convert_tflite.py
import tensorflow as tf

def convert(h5_path="models/baby_cry_model_final.h5", tflite_path="models/baby_cry_model.tflite"):
    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optional: post-training quantization for smaller size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("Saved tflite to", tflite_path)

if __name__ == "__main__":
    convert()
