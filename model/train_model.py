import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

print("="*60)
print("AGRICULTURAL DISEASE DETECTION SYSTEM - TRAINING")
print("="*60)

def resolve_image_path(base_dir, image_path_value):
    """Resolve image path from CSV to a valid absolute path under base_dir.

    Handles cases where CSV stores paths like 'color\\Class\\file.JPG' while
    base_dir already points to the 'color' directory, which would otherwise
    create '...\\color\\color\\...'.
    """
    # Normalize separators
    raw_path = str(image_path_value).replace('/', os.sep).replace('\\', os.sep)

    # Absolute path in CSV: return as-is
    if os.path.isabs(raw_path):
        return os.path.normpath(raw_path)

    # Drop leading './' or '.\\'
    if raw_path.startswith('.' + os.sep):
        raw_path = raw_path[2:]

    # If the relative path already starts with 'color\\', strip it
    # because base_dir is the absolute path to '...\\code_cortex\\color'
    lower_path = raw_path.lower()
    color_prefix = 'color' + os.sep
    if lower_path.startswith(color_prefix):
        raw_path = raw_path[len(color_prefix):]

    # Build candidate paths and return the first that exists
    project_root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.normpath(os.path.join(base_dir, raw_path)),
        os.path.normpath(os.path.join(project_root, 'color', raw_path)),
        os.path.normpath(os.path.join(project_root, raw_path)),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # Fallback to the first candidate if none exist (will error later and be logged)
    return candidates[0]

def check_environment():
    """Check if required packages are installed"""
    print("\n1. CHECKING ENVIRONMENT...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'sklearn': 'scikit-learn',
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✓ {package} imported successfully")
        except ImportError:
            print(f"   ✗ {package} missing - install with: pip install {pip_name}")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\nERROR: Missing packages. Run:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("   ✓ All required packages available!")
    return True

def check_files():
    """Check if required files exist"""
    print("\n2. CHECKING FILES...")
    
    csv_path = r"C:\Users\rajveersinh chavda\Documents\Projects\final\plant_disease_multimodal_dataset.csv"
    img_path = r"C:\Users\rajveersinh chavda\Documents\Projects\final"
    
    print(f"   Checking CSV: {csv_path}")
    csv_exists = os.path.exists(csv_path)
    print(f"   CSV file exists: {csv_exists}")
    
    print(f"   Checking images: {img_path}")
    img_exists = os.path.exists(img_path)
    print(f"   Image directory exists: {img_exists}")
    
    if csv_exists:
        try:
            df = pd.read_csv(csv_path)
            print(f"   ✓ CSV loaded: {len(df)} rows, {len(df.columns)} columns")
            print(f"   Columns: {list(df.columns)}")
            
            required_cols = ['K', 'temperature', 'humidity', 'ph', 'rainfall', 'Label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"   ✗ Missing columns: {missing_cols}")
                return False, None, None
            else:
                print(f"   ✓ All required columns present!")
                return True, csv_path, img_path
                
        except Exception as e:
            print(f"   ✗ Error reading CSV: {e}")
            return False, None, None
    else:
        print(f"   ✗ CSV file not found!")
        return False, None, None

def train_disease_risk_model(csv_path):
    """Train Disease Risk Prediction Model"""
    print("\n3. TRAINING DISEASE RISK PREDICTION MODEL...")
    print("-" * 50)
    
    try:
        # Load data
        df = pd.read_csv(csv_path)
        
        # Use sample for faster training (remove this line for full dataset)
        df = df.sample(n=50000, random_state=42)
        print(f"   Using sample dataset: {len(df)} samples")
        
        # Prepare features and labels
        feature_cols = ['K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = df[feature_cols].copy()
        y = df['Label'].copy()
        
        print(f"   Features: {feature_cols}")
        print(f"   Unique diseases: {y.nunique()}")
        print(f"   Disease distribution:")
        for disease, count in y.value_counts().head().items():
            print(f"     {disease}: {count}")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print(f"   Encoded {len(label_encoder.classes_)} disease classes")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("   Features scaled successfully")
        
        # Train model with faster parameters
        print("   Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=50,    # Reduced for faster training
            random_state=42, 
            max_depth=8,        # Slightly reduced depth
            min_samples_split=2,
            n_jobs=-1           # Use all CPU cores
        )
        
        model.fit(X_train_scaled, y_train)
        print("   Model training completed!")
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Feature importance
        importance = dict(zip(feature_cols, model.feature_importances_))
        print("   Feature importance:")
        for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"     {feature}: {imp:.4f}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/disease_risk_model.pkl')
        joblib.dump(scaler, 'models/risk_scaler.pkl')
        joblib.dump(label_encoder, 'models/risk_label_encoder.pkl')
        
        # Save metadata
        metadata = {
            'model_type': 'Disease Risk Prediction',
            'feature_columns': feature_cols,
            'disease_classes': list(label_encoder.classes_),
            'accuracy': float(accuracy),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        with open('models/risk_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("   ✓ Disease Risk Model saved successfully!")
        return True, model, scaler, label_encoder
        
    except Exception as e:
        print(f"   ✗ Error training risk model: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None

def train_image_detection_model(csv_path, img_path):
    """Train Disease Detection Model (Image-based)"""
    print("\n4. TRAINING DISEASE DETECTION MODEL (IMAGE-BASED)...")
    print("-" * 50)
    
    try:
        # Check for required packages
        try:
            import cv2
            print("   ✓ OpenCV available")
        except ImportError:
            print("   ✗ OpenCV not available, skipping image model")
            return False, None, None
        
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            from tensorflow.keras.applications import EfficientNetB0
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            print("   ✓ TensorFlow available")
        except ImportError:
            print("   ✗ TensorFlow not available, skipping image model")
            return False, None, None
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Check for Image Path column (with space, not underscore)
        if 'Image Path' not in df.columns:
            print("   ✗ No 'Image Path' column found in CSV")
            print(f"   Available columns: {list(df.columns)}")
            return False, None, None
        
        print(f"   ✓ Found 'Image Path' column")
        
        # Use sample for faster training
        df_sample = df.sample(n=5000, random_state=42)
        print(f"   Using sample dataset: {len(df_sample)} images")
        
        # Load and preprocess images
        print("   Loading and preprocessing images...")
        images = []
        labels = []
        failed_count = 0
        
        # Show a couple of resolved path samples for quick diagnostics
        sample_debug_printed = 0

        for idx, row in df_sample.iterrows():
            try:
                # Use 'Image Path' with space, not underscore
                img_relative_path = row['Image Path']
                img_full_path = resolve_image_path(img_path, img_relative_path)
                if sample_debug_printed < 3:
                    print(f"     Resolved: '{img_relative_path}' -> '{img_full_path}' | Exists: {os.path.exists(img_full_path)}")
                    sample_debug_printed += 1
                
                img = cv2.imread(img_full_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    img = img.astype('float32') / 255.0
                    images.append(img)
                    labels.append(row['Label'])
                else:
                    failed_count += 1
                    if failed_count <= 5:  # Show only first 5 failures
                        print(f"     Warning: Could not load {img_full_path}")
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:
                    print(f"     Error loading {row['Image Path']}: {e}")
        
        if len(images) == 0:
            print("   ✗ No images loaded successfully")
            return False, None, None
        
        print(f"   ✓ Loaded {len(images)} images successfully ({failed_count} failed)")
        
        # Convert to numpy arrays
        X_img = np.array(images)
        y_img = np.array(labels)
        
        # Encode labels
        img_label_encoder = LabelEncoder()
        y_img_encoded = img_label_encoder.fit_transform(y_img)
        y_img_categorical = tf.keras.utils.to_categorical(y_img_encoded)
        
        print(f"   Disease classes for images: {len(img_label_encoder.classes_)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_img, y_img_categorical, test_size=0.2, random_state=42, stratify=y_img_encoded
        )
        
        print(f"   Image train samples: {len(X_train)}, test samples: {len(X_test)}")
        
        # Create model
        def create_disease_detection_model(num_classes):
            base_model = EfficientNetB0(
                weights='imagenet', 
                include_top=False, 
                input_shape=(224, 224, 3)
            )
            base_model.trainable = False
            
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        
        num_classes = len(img_label_encoder.classes_)
        img_model = create_disease_detection_model(num_classes)
        
        print("   Training image detection model...")
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=5, monitor='val_accuracy', restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=3, monitor='val_loss'
            )
        ]
        
        # Train model
        history = img_model.fit(
            train_datagen.flow(X_train, y_train, batch_size=32),
            validation_data=(X_test, y_test),
            epochs=10,  # Reduced for faster training
            callbacks=callbacks,
            verbose=1
        )
        
        # Save image model
        img_model.save('models/disease_detection_model.h5')
        joblib.dump(img_label_encoder, 'models/detection_label_encoder.pkl')
        
        # Save image model metadata
        img_metadata = {
            'model_type': 'Disease Detection (Image-based)',
            'num_classes': num_classes,
            'disease_classes': list(img_label_encoder.classes_),
            'input_shape': [224, 224, 3],
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        with open('models/detection_model_metadata.json', 'w') as f:
            json.dump(img_metadata, f, indent=2)
        
        print("   ✓ Disease Detection Model saved successfully!")
        return True, img_model, img_label_encoder
        
    except Exception as e:
        print(f"   ✗ Error training image model: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def predict_disease_risk(soil_data, model_path='models'):
    """
    Predict disease risk from soil/climate data
    Returns API-ready JSON format
    """
    try:
        model = joblib.load(f'{model_path}/disease_risk_model.pkl')
        scaler = joblib.load(f'{model_path}/risk_scaler.pkl')
        label_encoder = joblib.load(f'{model_path}/risk_label_encoder.pkl')

        input_data = np.array([[
            soil_data['K'], soil_data['temperature'], soil_data['humidity'],
            soil_data['ph'], soil_data['rainfall']
        ]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        # Get disease name and confidence
        predicted_disease = label_encoder.inverse_transform([prediction])[0]
        confidence = float(max(probabilities))

        # Create probability dictionary
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            disease_name = label_encoder.inverse_transform([i])[0]
            # Simplify disease names for API (remove crop prefix)
            simple_name = disease_name.split('___')[-1] if '___' in disease_name else disease_name
            prob_dict[simple_name] = round(float(prob), 4)

        # Determine risk level
        if confidence > 0.7:
            risk_level = "High"
        elif confidence > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            'disease_risk': risk_level,
            'probabilities': prob_dict
        }
        
    except Exception as e:
        return {"error": f"Error predicting disease risk: {e}"}

def predict_disease_from_image(image_path, model_path='models'):
    """
    Predict disease from image
    Returns API-ready JSON format
    """
    try:
        import cv2
        import tensorflow as tf
        
        model = tf.keras.models.load_model(f'{model_path}/disease_detection_model.h5')
        label_encoder = joblib.load(f'{model_path}/detection_label_encoder.pkl')

        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        predictions = model.predict(img)[0]
        
        # Get top prediction
        predicted_class_idx = np.argmax(predictions)
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = float(predictions[predicted_class_idx])
        
        # Simplify disease name for API
        simple_disease = predicted_class.split('___')[-1] if '___' in predicted_class else predicted_class
        
        # Create confidence scores for all classes
        all_confidences = {}
        for i, prob in enumerate(predictions):
            class_name = label_encoder.inverse_transform([i])[0]
            simple_name = class_name.split('___')[-1] if '___' in class_name else class_name
            all_confidences[simple_name] = round(float(prob), 4)
        
        return {
            "disease": simple_disease,
            "confidence": round(confidence, 4),
            "all_confidences": all_confidences
        }
        
    except Exception as e:
        return {"error": f"Error processing image: {e}"}

def test_predictions(risk_model, scaler, label_encoder):
    """Test both models with sample predictions"""
    print("\n5. TESTING TRAINED MODELS...")
    print("-" * 50)
    
    try:
        # Test Disease Risk Prediction
        print("   Testing Disease Risk Prediction...")
        test_soil_data = {
            'K': 30, 'temperature': 25.0, 'humidity': 65, 
            'ph': 6.5, 'rainfall': 180
        }
        
        risk_result = predict_disease_risk(test_soil_data)
        print(f"   Risk Prediction Result:")
        print(f"   {json.dumps(risk_result, indent=4)}")
        
        # Test Disease Detection if model exists
        if os.path.exists('models/disease_detection_model.h5'):
            print("\n   Disease Detection model available for testing")
            print("   Example output format:")
            example_detection = {
                "disease": "Late_Blight",
                "confidence": 0.93,
                "all_confidences": {
                    "Late_Blight": 0.93,
                    "Healthy": 0.07
                }
            }
            print(f"   {json.dumps(example_detection, indent=4)}")
        
        print("   ✓ Model testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ✗ Error testing models: {e}")
        return False

def main():
    """Main training function"""
    print("Starting Agricultural Disease Detection Training...")
    
    # Step 1: Check environment
    if not check_environment():
        print("\nERROR: Environment setup failed!")
        return
    
    # Step 2: Check files
    files_ok, csv_path, img_path = check_files()
    if not files_ok:
        print("\nERROR: Required files not found!")
        return
    
    # Step 3: Train risk prediction model
    risk_success, model, scaler, label_encoder = train_disease_risk_model(csv_path)
    if not risk_success:
        print("\nERROR: Risk model training failed!")
        return
    
    # Step 4: Train image detection model
    img_success, img_model, img_label_encoder = train_image_detection_model(csv_path, img_path)
    
    # Step 5: Test predictions
    if model and scaler and label_encoder:
        test_predictions(model, scaler, label_encoder)
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"✓ Disease Risk Prediction Model: {'SUCCESS' if risk_success else 'FAILED'}")
    print(f"✓ Disease Detection Model: {'SUCCESS' if img_success else 'FAILED'}")
    print(f"✓ Models saved in: {os.path.abspath('models')}")
    print("="*60)
    
    if risk_success and img_success:
        print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        print("Both disease risk prediction and image detection models are ready!")
    elif risk_success:
        print("\n✅ DISEASE RISK MODEL READY!")
        print("Image detection model needs OpenCV/TensorFlow or more images.")
    
    print("\nAPI-Ready Output Formats:")
    print("- Disease Risk: {'disease_risk': 'High', 'probabilities': {...}}")
    print("- Disease Detection: {'disease': 'Late_Blight', 'confidence': 0.93}")

if __name__ == "__main__":
    main()
