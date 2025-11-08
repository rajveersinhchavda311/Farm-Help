# model/predict_disease.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import argparse
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, 'disease_detection_model.h5')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'detection_label_encoder.pkl')

try:
    # Try loading with compile=False to avoid compatibility issues
    disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH, compile=False)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(json.dumps({"error": f"Error loading image model or encoder: {str(e)}"}))
    sys.exit(1)

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    args = parser.parse_args()

    try:
        preprocessed_img = preprocess_image(args.image_path)
        predictions = disease_model.predict(preprocessed_img)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        predicted_disease = label_encoder.inverse_transform([predicted_class_idx])[0]
        all_confidences = {label_encoder.classes_[i]: float(predictions[0][i]) for i in range(len(predictions[0]))}
        result = {"disease": predicted_disease, "confidence": confidence, "all_confidences": all_confidences}
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()