import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"model path"
model = tf.keras.models.load_model(model_path)

class_indices = json.load(open(f"{working_dir}/class_indices.json"))


def load_and_process_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255
    return img_array


def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_process_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name
