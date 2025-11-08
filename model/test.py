import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import cv2
import tensorflow as tf
from glob import glob


print("="*60)
print("AGRICULTURAL DISEASE DETECTION - MODEL TESTING")
print("="*60)


def resolve_image_path(base_dir, image_path_value):
    """Resolve image path from CSV to a valid absolute path"""
    raw_path = str(image_path_value).replace('/', os.sep).replace('\\', os.sep)
    
    if os.path.isabs(raw_path):
        return os.path.normpath(raw_path)
    
    if raw_path.startswith('.' + os.sep):
        raw_path = raw_path[2:]
    lower_path = raw_path.lower()
    color_prefix = 'color' + os.sep
    if lower_path.startswith(color_prefix):
        raw_path = raw_path[len(color_prefix):]
    project_root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.normpath(os.path.join(base_dir, raw_path)),
        os.path.normpath(os.path.join(project_root, 'color', raw_path)),
        os.path.normpath(os.path.join(project_root, raw_path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates


def check_models():
    """Check if trained models exist"""
    print("\n1. CHECKING TRAINED MODELS")
    model_files = {
        'Disease Risk Model': 'models/disease_risk_model.pkl',
        'Risk Scaler': 'models/risk_scaler.pkl',
        'Risk Label Encoder': 'models/risk_label_encoder.pkl',
        'Disease Detection Model': 'models/disease_detection_model.h5',
        'Detection Label Encoder': 'models/detection_label_encoder.pkl'
    }
    available_models = {}
    for model_name, file_path in model_files.items():
        exists = os.path.exists(file_path)
        print(f"   {model_name}: {'✓' if exists else '✗'} {file_path}")
        available_models[model_name] = exists
    # Check metadata files
    risk_meta = os.path.exists('models/risk_model_metadata.json')
    detection_meta = os.path.exists('models/detection_model_metadata.json')
    print(f"   Risk Model Metadata: {'✓' if risk_meta else '✗'}")
    print(f"   Detection Model Metadata: {'✓' if detection_meta else '✗'}")
    return available_models


def load_metadata():
    """Load and display model metadata"""
    print("\n2. LOADING MODEL METADATA")
    metadata = {}
    if os.path.exists('models/risk_model_metadata.json'):
        with open('models/risk_model_metadata.json') as f:
            risk_meta = json.load(f)
            metadata['risk'] = risk_meta
            print(f"   Risk Model Accuracy: {risk_meta.get('accuracy', 'N/A'):.4f}")
            print(f"   Disease Classes: {len(risk_meta.get('disease_classes', []))}")
            print(f"   Features: {risk_meta.get('feature_columns', [])}")
    if os.path.exists('models/detection_model_metadata.json'):
        with open('models/detection_model_metadata.json') as f:
            detection_meta = json.load(f)
            metadata['detection'] = detection_meta
            print(f"   Detection Classes: {len(detection_meta.get('disease_classes', []))}")
            print(f"   Input Shape: {detection_meta.get('input_shape', [])}")
            print(f"   Training Samples: {detection_meta.get('training_samples', 'N/A')}")
    return metadata


def test_risk_prediction():
    """Test Disease Risk Prediction Model"""
    print("\n3. TESTING DISEASE RISK PREDICTION MODEL")
    try:
        model = joblib.load('models/disease_risk_model.pkl')
        scaler = joblib.load('models/risk_scaler.pkl')
        label_encoder = joblib.load('models/risk_label_encoder.pkl')
        print("   Model components loaded successfully")
        print(f"   Available classes: {len(label_encoder.classes_)}")
        test_cases = [
            {'name': 'High Risk', 'data': {'K': 15, 'temperature': 32.0, 'humidity': 85, 'ph': 5.0, 'rainfall': 280}},
            {'name': 'Moderate Risk', 'data': {'K': 30, 'temperature': 25.0, 'humidity': 65, 'ph': 6.5, 'rainfall': 180}},
            {'name': 'Low Risk', 'data': {'K': 45, 'temperature': 20.0, 'humidity': 50, 'ph': 7.0, 'rainfall': 120}},
            {'name': 'Extreme', 'data': {'K': 10, 'temperature': 35.0, 'humidity': 95, 'ph': 4.5, 'rainfall': 350}},
            {'name': 'Optimal', 'data': {'K': 40, 'temperature': 22.0, 'humidity': 60, 'ph': 6.8, 'rainfall': 150}}
        ]
        results = []
        for i, test_case in enumerate(test_cases):
            print(f"\n   Test {i+1}: {test_case['name']}")
            print(f"   Input: {test_case['data']}")
            try:
                input_arr = np.array([[test_case['data']['K'], test_case['data']['temperature'], test_case['data']['humidity'], test_case['data']['ph'], test_case['data']['rainfall']]])
                scaled = scaler.transform(input_arr)
                prediction = model.predict(scaled)[0]
                probabilities = model.predict_proba(scaled)[0]
                predicted = label_encoder.inverse_transform([prediction])[0]
                confidence = float(max(probabilities))
                prob_dict = {}
                for j, prob in enumerate(probabilities):
                    disease = label_encoder.inverse_transform([j])[0]
                    simple_name = disease.split('___')[-1] if '___' in disease else disease
                    prob_dict[simple_name] = round(float(prob), 4)
                prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
                top3 = dict(list(prob_dict.items())[:3])
                if confidence > 0.7:
                    risk_level = "High"
                elif confidence > 0.4:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
                result = {'risk': risk_level, 'probabilities': top3}
                print(f"   Output: {json.dumps(result, indent=4)}")
                results.append({'test': test_case['name'], 'input': test_case['data'], 'output': result})
            except Exception as e:
                print(f"   Prediction error: {e}")
        print("\n   Risk prediction testing done")
        return True, results
    except Exception as e:
        print(f"   Loading error: {e}")
        return False, []


def test_detection_model():
    """Test Disease Detection Model"""
    print("\n4. TESTING DISEASE DETECTION MODEL")
    try:
        if not os.path.exists('models/disease_detection_model.h5'):
            print("   Detection model missing")
            return False, []
        model = tf.keras.models.load_model('models/disease_detection_model.h5')
        label_encoder = joblib.load('models/detection_label_encoder.pkl')
        print("   Model components loaded")
        print(f"   Classes: {len(label_encoder.classes_)}")
        csv_path = r"C:\Users\rajveersinh chavda\Documents\plant_multimodal.csv"
        base_dir = r"C:\Users\rajveersinh chavda\Documents"
        if not os.path.exists(csv_path):
            print("   CSV missing")
            return False, []
        df = pd.read_csv(csv_path)
        samples = []
        for label in df['Label'].unique()[:5]:
            sample_df = df[df['Label'] == label]
            if not sample_df.empty:
                samples.append({'path': sample_df.iloc[0]['Image Path'], 'label': label})
        print(f"   Testing {len(samples)} samples")
        results = []
        for i, sample in enumerate(samples):
            print(f"\n   Sample {i+1}: {sample['label']}")
            img_path = resolve_image_path(base_dir, sample['path'])
            print(f"   Image Path: {img_path}")
            if not os.path.exists(img_path):
                print("   Image not found")
                continue
            img = cv2.imread(img_path)
            if img is None:
                print("   Failed to read image")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            pred = model.predict(img, verbose=0)[0]
            pred_idx = np.argmax(pred)
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(pred[pred_idx])
            simple_label = pred_label.split('___')[-1] if '___' in pred_label else pred_label
            top_idxs = np.argsort(pred)[::-1][:5]
            confs = {label_encoder.inverse_transform([idx])[0].split('___')[-1]: round(float(pred[idx]), 4) for idx in top_idxs}
            output = {'disease': simple_label, 'confidence': confidence, 'top_confidences': confs}
            print(f"   Output: {json.dumps(output, indent=4)}")
            correct = simple_label.lower() == sample['label'].split('___')[-1].lower()
            print(f"   Correct: {'Yes' if correct else 'No'}")
            results.append({'image': sample['path'], 'correct': correct, 'output': output})
        print("\n   Detection testing done")
        correct_total = sum(r['correct'] for r in results)
        print(f"   Accuracy: {100 * correct_total / len(results) if results else 0:.2f}%")
        return True, results
    except Exception as e:
        print(f"   Detection test error: {e}")
        return False, []


def test_api_functions():
    """Test API-ready functions"""
    print("\n5. TESTING API READY FUNCTIONS")
    try:
        from train_model import predict_disease_risk, predict_disease_from_image
        print("   Testing disease risk prediction")
        test_soil = {'K': 25, 'temperature': 28.0, 'humidity':70}
