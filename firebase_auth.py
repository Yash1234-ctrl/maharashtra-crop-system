from firebase_config import FirebaseManager
from datetime import datetime, timedelta

class FirebaseAuthDB:
    def __init__(self):
        """Initialize Firebase authentication manager"""
        self.firebase = FirebaseManager()
    
    async def register_farmer(self, username, email, password, full_name, **kwargs):
        """Register a new farmer account using Firebase Authentication"""
        try:
            # Create user in Firebase
            result = await self.firebase.sign_up_user(
                email=email,
                password=password,
                full_name=full_name,
                phone=kwargs.get('phone', ''),
                location={
                    'district': kwargs.get('district', ''),
                    'village': kwargs.get('village', '')
                }
            )
            
            if result['success']:
                # Store additional profile data
                profile_data = {
                    'username': username,
                    'farm_name': kwargs.get('farm_name', ''),
                    'farm_area': kwargs.get('farm_area', 0),
                    'crop_types': kwargs.get('crop_types', '').split(','),
                    'profile_picture': kwargs.get('profile_picture', ''),
                    'registration_date': datetime.now(),
                    'last_login': None
                }
                
                # Update user profile in Firestore
                await self.firebase.update_user_profile(result['user_id'], profile_data)
                
                return {
                    'success': True,
                    'message': 'Farmer registered successfully!',
                    'farmer_id': result['user_id']
                }
            else:
                return {'success': False, 'message': result.get('error', 'Registration failed')}
                
        except Exception as e:
            return {'success': False, 'message': f'Registration failed: {str(e)}'}
    
    async def authenticate_farmer(self, email, password, ip_address=None):
        """Authenticate farmer using Firebase Authentication"""
        try:
            result = await self.firebase.sign_in_user(email, password)
            
            if result['success']:
                # Update last login timestamp
                await self.firebase.update_user_profile(
                    result['user_id'],
                    {'last_login': datetime.now()}
                )
                
                # Create session token
                session = await self.create_session(
                    result['user_id'],
                    ip_address=ip_address
                )
                
                if session['success']:
                    return {
                        'success': True,
                        'message': 'Login successful!',
                        'farmer_id': result['user_id'],
                        'full_name': result['user_data'].get('full_name', ''),
                        'session': session
                    }
                else:
                    return {'success': False, 'message': 'Session creation failed'}
            else:
                return {'success': False, 'message': result.get('error', 'Authentication failed')}
                
        except Exception as e:
            return {'success': False, 'message': f'Authentication failed: {str(e)}'}
    
    async def create_session(self, farmer_id, ip_address=None):
        """Create a new session in Firebase"""
        try:
            # Generate custom token
            custom_token = await self.firebase.create_custom_token(farmer_id)
            
            session_data = {
                'farmer_id': farmer_id,
                'token': custom_token,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(days=7),
                'ip_address': ip_address,
                'is_active': True
            }
            
            # Store session in Firestore
            result = await self.firebase.create_session(session_data)
            
            if result['success']:
                return {
                    'success': True,
                    'session_id': result['session_id'],
                    'token': custom_token,
                    'expires_at': session_data['expires_at'].isoformat()
                }
            else:
                return {'success': False, 'message': 'Failed to create session'}
                
        except Exception as e:
            return {'success': False, 'message': f'Session creation failed: {str(e)}'}
    
    async def validate_session(self, session_id, token):
        """Validate farmer session in Firebase"""
        try:
            result = await self.firebase.verify_session_token(session_id, token)
            
            if result['success']:
                return {
                    'success': True,
                    'farmer_id': result['farmer_id'],
                    'user_data': result['user_data']
                }
            else:
                return {'success': False, 'message': 'Invalid or expired session'}
                
        except Exception as e:
            return {'success': False, 'message': f'Session validation failed: {str(e)}'}
    
    async def invalidate_session(self, session_id):
        """Invalidate a session in Firebase"""
        try:
            result = await self.firebase.revoke_session(session_id)
            return result
        except Exception as e:
            return {'success': False, 'message': f'Session invalidation failed: {str(e)}'}
    
    async def get_farmer_profile(self, farmer_id):
        """Get farmer profile from Firebase"""
        try:
            result = await self.firebase.get_user_profile(farmer_id)
            
            if result['success']:
                return {
                    'success': True,
                    'profile': result['profile']
                }
            else:
                return {'success': False, 'message': 'Profile not found'}
                
        except Exception as e:
            return {'success': False, 'message': f'Profile retrieval failed: {str(e)}'}
    
    async def update_farmer_profile(self, farmer_id, profile_data):
        """Update farmer profile in Firebase"""
        try:
            result = await self.firebase.update_user_profile(farmer_id, profile_data)
            return result
        except Exception as e:
            return {'success': False, 'message': f'Profile update failed: {str(e)}'}
    
    async def get_farmer_preferences(self, farmer_id):
        """Get farmer preferences from Firebase"""
        try:
            result = await self.firebase.get_user_preferences(farmer_id)
            return result
        except Exception as e:
            return {'success': False, 'message': f'Preferences retrieval failed: {str(e)}'}
    
    async def update_farmer_preferences(self, farmer_id, preferences):
        """Update farmer preferences in Firebase"""
        try:
            result = await self.firebase.update_user_preferences(farmer_id, preferences)
            return result
        except Exception as e:
            return {'success': False, 'message': f'Preferences update failed: {str(e)}'}
    
    async def get_farmer_history(self, farmer_id):
        """Get farmer's analysis history from Firebase"""
        try:
            result = await self.firebase.get_user_history(farmer_id)
            return result
        except Exception as e:
            return {'success': False, 'message': f'History retrieval failed: {str(e)}'}