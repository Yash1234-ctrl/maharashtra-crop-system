import firebase_admin
from firebase_admin import credentials, firestore, storage, auth
import pyrebase

# Firebase Admin SDK configuration
cred = credentials.Certificate("path/to/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'your-bucket-name.appspot.com'
})

# Pyrebase configuration for client-side operations
firebase_config = {
    'apiKey': "your-api-key",
    'authDomain': "your-project-id.firebaseapp.com",
    'projectId': "your-project-id",
    'storageBucket': "your-project-id.appspot.com",
    'messagingSenderId': "your-sender-id",
    'appId': "your-app-id",
    'databaseURL': "your-database-url",
    'measurementId': "your-measurement-id"
}

# Initialize Pyrebase
firebase = pyrebase.initialize_app(firebase_config)
pyrebase_auth = firebase.auth()
pyrebase_storage = firebase.storage()

# Get Firestore instance
db = firestore.client()

class FirebaseManager:
    @staticmethod
    async def sign_up_user(email, password, full_name, phone, location):
        try:
            # Create user in Firebase Authentication
            user = auth.create_user(
                email=email,
                password=password,
                display_name=full_name
            )
            
            # Store additional user data in Firestore
            user_ref = db.collection('users').document(user.uid)
            user_ref.set({
                'full_name': full_name,
                'email': email,
                'phone': phone,
                'location': location,
                'created_at': firestore.SERVER_TIMESTAMP,
                'role': 'farmer'
            })
            
            return {'success': True, 'user_id': user.uid}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    async def sign_in_user(email, password):
        try:
            # Sign in with Pyrebase
            user = pyrebase_auth.sign_in_with_email_and_password(email, password)
            
            # Get additional user data from Firestore
            user_ref = db.collection('users').document(user['localId'])
            user_data = user_ref.get().to_dict()
            
            return {
                'success': True,
                'user_id': user['localId'],
                'token': user['idToken'],
                'user_data': user_data
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    async def save_crop_analysis(user_id, analysis_data):
        try:
            # Store analysis in Firestore
            analysis_ref = db.collection('crop_analysis').document()
            analysis_ref.set({
                'user_id': user_id,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'data': analysis_data
            })
            return {'success': True, 'analysis_id': analysis_ref.id}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    async def save_soil_analysis(user_id, soil_data):
        try:
            # Store soil analysis in Firestore
            soil_ref = db.collection('soil_analysis').document()
            soil_ref.set({
                'user_id': user_id,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'data': soil_data
            })
            return {'success': True, 'analysis_id': soil_ref.id}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    async def upload_image(user_id, image_file, image_type):
        try:
            # Upload to Firebase Storage
            file_path = f"images/{user_id}/{image_type}/{image_file.name}"
            blob = pyrebase_storage.child(file_path).put(image_file)
            url = pyrebase_storage.child(file_path).get_url(None)
            
            return {'success': True, 'url': url, 'path': file_path}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    async def get_user_history(user_id):
        try:
            # Get crop analysis history
            crop_analyses = db.collection('crop_analysis')\
                            .where('user_id', '==', user_id)\
                            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                            .limit(10)\
                            .stream()
            
            # Get soil analysis history
            soil_analyses = db.collection('soil_analysis')\
                            .where('user_id', '==', user_id)\
                            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                            .limit(10)\
                            .stream()
            
            return {
                'success': True,
                'crop_analyses': [doc.to_dict() for doc in crop_analyses],
                'soil_analyses': [doc.to_dict() for doc in soil_analyses]
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}