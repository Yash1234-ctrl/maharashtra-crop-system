from firebase_functions import https_fn
from firebase_admin import initialize_app, firestore
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import base64

# Initialize Firebase
app = initialize_app()
db = firestore.client()

# Load models
try:
    model = tf.keras.models.load_model('best_model.h5')
except:
    model = None

@https_fn.on_request()
def predict_crop(req: https_fn.Request) -> https_fn.Response:
    """Main endpoint for crop prediction and analysis"""
    try:
        # Get data from request
        data = req.get_json()
        if not data:
            return https_fn.Response(json={
                "error": "No data provided",
                "status": "error"
            }, status=400)

        # Handle image analysis if image is provided
        if 'image' in data:
            image_data = base64.b64decode(data['image'])
            results = analyze_image(image_data)
        else:
            results = {
                "message": "No image provided for analysis",
                "status": "warning"
            }

        # Save analysis to Firestore
        doc_ref = db.collection('crop_analysis').add({
            'timestamp': firestore.SERVER_TIMESTAMP,
            'results': results,
            'metadata': {
                'district': data.get('district', ''),
                'crop_type': data.get('crop_type', ''),
                'farm_area': data.get('farm_area', 0)
            }
        })

        return https_fn.Response(json={
            "results": results,
            "status": "success",
            "analysis_id": doc_ref.id
        })

    except Exception as e:
        return https_fn.Response(json={
            "error": str(e),
            "status": "error"
        }, status=500)

def analyze_image(image_data):
    """Analyze crop image using ML model"""
    try:
        # Convert image data to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Process image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for model
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        
        if model is not None:
            # Make prediction
            prediction = model.predict(np.expand_dims(image_array, axis=0))[0]
            predicted_class = np.argmax(prediction)
            confidence = float(prediction[predicted_class])
            
            class_names = ['Healthy', 'Early_Blight', 'Late_Blight', 'Bacterial_Spot']
            disease = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"
            
            return {
                'disease': disease,
                'confidence': round(confidence * 100, 2),
                'analysis_type': 'ML Model Prediction'
            }
        else:
            # Fallback to basic image analysis
            return {
                'analysis_type': 'Basic Analysis',
                'message': 'ML model not available, performed basic image analysis',
                'image_quality': 'Good' if np.mean(image_array) > 0.3 else 'Poor'
            }
            
    except Exception as e:
        return {
            'error': str(e),
            'analysis_type': 'Failed',
            'status': 'error'
        }