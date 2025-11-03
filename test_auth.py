#!/usr/bin/env python3
"""Test MongoDB Authentication System"""

from mongodb_auth import MongoFarmerAuth

def test_auth_system():
    """Test the MongoDB authentication system"""
    # Initialize auth system
    auth = MongoFarmerAuth()
    
    # Test farmer registration
    test_farmer = {
        'username': 'test_farmer',
        'email': 'test@example.com',
        'password': 'Test@123',
        'full_name': 'Test Farmer',
        'phone': '1234567890',
        'farm_name': 'Test Farm',
        'district': 'Pune',
        'village': 'Test Village',
        'farm_area': 5.0,
        'crop_types': 'Cotton,Wheat'
    }
    
    print("\n1. Testing Farmer Registration...")
    reg_result = auth.register_farmer(
        username=test_farmer['username'],
        email=test_farmer['email'],
        password=test_farmer['password'],
        full_name=test_farmer['full_name'],
        phone=test_farmer['phone'],
        farm_name=test_farmer['farm_name'],
        district=test_farmer['district'],
        village=test_farmer['village'],
        farm_area=test_farmer['farm_area'],
        crop_types=test_farmer['crop_types']
    )
    print(f"Registration Result: {reg_result}")
    
    if reg_result['success']:
        farmer_id = reg_result['farmer_id']
        
        print("\n2. Testing Authentication...")
        auth_result = auth.authenticate_farmer(
            username=test_farmer['username'],
            password=test_farmer['password'],
            ip_address='127.0.0.1'
        )
        print(f"Authentication Result: {auth_result}")
        
        if auth_result['success']:
            print("\n3. Testing Session Creation...")
            session_result = auth.create_session(
                farmer_id=farmer_id,
                ip_address='127.0.0.1',
                user_agent='Test Browser'
            )
            print(f"Session Creation Result: {session_result}")
            
            if session_result['success']:
                print("\n4. Testing Session Validation...")
                validation_result = auth.validate_session(
                    session_id=session_result['session_id'],
                    session_token=session_result['session_token']
                )
                print(f"Session Validation Result: {validation_result}")
                
                print("\n5. Testing Profile Retrieval...")
                profile_result = auth.get_farmer_profile(farmer_id)
                print(f"Profile Retrieval Result: {profile_result}")
                
                print("\n6. Testing Profile Update...")
                update_result = auth.update_farmer_profile(
                    farmer_id=farmer_id,
                    updates={'farm_area': 6.0, 'crop_types': 'Cotton,Wheat,Sugarcane'}
                )
                print(f"Profile Update Result: {update_result}")
                
                print("\n7. Testing Session Invalidation...")
                logout_result = auth.invalidate_session(session_result['session_id'])
                print(f"Logout Result: {logout_result}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_auth_system()