#!/usr/bin/env python3
"""
Farmer Authentication Database
Secure user management system for Maharashtra Agricultural System
"""

import sqlite3
import hashlib
import secrets
import os
from datetime import datetime, timedelta
import bcrypt

class FarmerAuthDB:
    def __init__(self, db_path="farmer_auth.db"):
        """Initialize the farmer authentication database"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create database tables for farmer authentication"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Farmers table for user accounts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS farmers (
                    farmer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    phone TEXT,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    farm_name TEXT,
                    district TEXT,
                    village TEXT,
                    farm_area REAL,
                    crop_types TEXT,
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    profile_picture TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Sessions table for login management
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS farmer_sessions (
                    session_id TEXT PRIMARY KEY,
                    farmer_id INTEGER NOT NULL,
                    session_token TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (farmer_id) REFERENCES farmers (farmer_id)
                )
            """)
            
            # Login attempts table for security
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS login_attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    ip_address TEXT,
                    success BOOLEAN,
                    attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_message TEXT
                )
            """)
            
            # Farmer preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS farmer_preferences (
                    preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    farmer_id INTEGER NOT NULL,
                    language TEXT DEFAULT 'en',
                    theme TEXT DEFAULT 'agricultural',
                    notifications_enabled BOOLEAN DEFAULT 1,
                    email_alerts BOOLEAN DEFAULT 1,
                    sms_alerts BOOLEAN DEFAULT 0,
                    weather_alerts BOOLEAN DEFAULT 1,
                    pest_alerts BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (farmer_id) REFERENCES farmers (farmer_id)
                )
            """)
            
            conn.commit()
            print("✅ Farmer authentication database initialized successfully!")
    
    def hash_password(self, password):
        """Securely hash password with bcrypt"""
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8'), salt.decode('utf-8')
    
    def verify_password(self, password, stored_hash):
        """Verify password against stored hash"""
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
    
    def register_farmer(self, username, email, password, full_name, **kwargs):
        """Register a new farmer account"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if username or email already exists
                cursor.execute("SELECT farmer_id FROM farmers WHERE username = ? OR email = ?", 
                             (username, email))
                if cursor.fetchone():
                    return {"success": False, "message": "Username or email already exists"}
                
                # Hash password
                password_hash, salt = self.hash_password(password)
                
                # Insert farmer record
                cursor.execute("""
                    INSERT INTO farmers (
                        username, email, password_hash, salt, full_name, 
                        phone, farm_name, district, village, farm_area, crop_types
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    username, email, password_hash, salt, full_name,
                    kwargs.get('phone', ''), kwargs.get('farm_name', ''),
                    kwargs.get('district', ''), kwargs.get('village', ''),
                    kwargs.get('farm_area', 0), kwargs.get('crop_types', '')
                ))
                
                farmer_id = cursor.lastrowid
                
                # Create default preferences
                cursor.execute("""
                    INSERT INTO farmer_preferences (farmer_id) VALUES (?)
                """, (farmer_id,))
                
                conn.commit()
                
                return {
                    "success": True, 
                    "message": "Farmer registered successfully!",
                    "farmer_id": farmer_id
                }
                
        except Exception as e:
            return {"success": False, "message": f"Registration failed: {str(e)}"}
    
    def authenticate_farmer(self, username, password, ip_address=None):
        """Authenticate farmer login"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get farmer record
                cursor.execute("""
                    SELECT farmer_id, username, password_hash, full_name, is_active
                    FROM farmers WHERE username = ? OR email = ?
                """, (username, username))
                
                farmer_record = cursor.fetchone()
                
                # Log login attempt
                cursor.execute("""
                    INSERT INTO login_attempts (username, ip_address, success, error_message)
                    VALUES (?, ?, ?, ?)
                """, (username, ip_address, False, ""))
                
                if not farmer_record:
                    cursor.execute("""
                        UPDATE login_attempts 
                        SET error_message = 'User not found'
                        WHERE attempt_id = last_insert_rowid()
                    """)
                    conn.commit()
                    return {"success": False, "message": "Invalid username or password"}
                
                farmer_id, db_username, password_hash, full_name, is_active = farmer_record
                
                if not is_active:
                    cursor.execute("""
                        UPDATE login_attempts 
                        SET error_message = 'Account deactivated'
                        WHERE attempt_id = last_insert_rowid()
                    """)
                    conn.commit()
                    return {"success": False, "message": "Account is deactivated"}
                
                # Verify password
                if not self.verify_password(password, password_hash):
                    cursor.execute("""
                        UPDATE login_attempts 
                        SET error_message = 'Invalid password'
                        WHERE attempt_id = last_insert_rowid()
                    """)
                    conn.commit()
                    return {"success": False, "message": "Invalid username or password"}
                
                # Update successful login
                cursor.execute("""
                    UPDATE login_attempts 
                    SET success = 1, error_message = 'Login successful'
                    WHERE attempt_id = last_insert_rowid()
                """)
                
                # Update last login time
                cursor.execute("""
                    UPDATE farmers SET last_login = CURRENT_TIMESTAMP WHERE farmer_id = ?
                """, (farmer_id,))
                
                conn.commit()
                
                return {
                    "success": True,
                    "message": "Login successful!",
                    "farmer_id": farmer_id,
                    "username": db_username,
                    "full_name": full_name
                }
                
        except Exception as e:
            return {"success": False, "message": f"Authentication failed: {str(e)}"}
    
    def create_session(self, farmer_id, ip_address=None, user_agent=None):
        """Create a new session for authenticated farmer"""
        session_id = secrets.token_urlsafe(32)
        session_token = secrets.token_urlsafe(64)
        expires_at = datetime.now() + timedelta(days=7)  # 7-day session
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO farmer_sessions 
                    (session_id, farmer_id, session_token, expires_at, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id, farmer_id, session_token, expires_at, ip_address, user_agent))
                
                conn.commit()
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "session_token": session_token,
                    "expires_at": expires_at.isoformat()
                }
                
        except Exception as e:
            return {"success": False, "message": f"Session creation failed: {str(e)}"}
    
    def validate_session(self, session_id, session_token):
        """Validate farmer session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT fs.farmer_id, fs.expires_at, f.username, f.full_name, f.district
                    FROM farmer_sessions fs
                    JOIN farmers f ON fs.farmer_id = f.farmer_id
                    WHERE fs.session_id = ? AND fs.session_token = ? 
                    AND fs.is_active = 1 AND fs.expires_at > CURRENT_TIMESTAMP
                """, (session_id, session_token))
                
                session_data = cursor.fetchone()
                
                if session_data:
                    farmer_id, expires_at, username, full_name, district = session_data
                    return {
                        "success": True,
                        "farmer_id": farmer_id,
                        "username": username,
                        "full_name": full_name,
                        "district": district,
                        "expires_at": expires_at
                    }
                else:
                    return {"success": False, "message": "Invalid or expired session"}
                    
        except Exception as e:
            return {"success": False, "message": f"Session validation failed: {str(e)}"}
    
    def invalidate_session(self, session_id):
        """Invalidate/logout a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE farmer_sessions 
                    SET is_active = 0 
                    WHERE session_id = ?
                """, (session_id,))
                
                conn.commit()
                return {"success": True, "message": "Session invalidated successfully"}
                
        except Exception as e:
            return {"success": False, "message": f"Session invalidation failed: {str(e)}"}
    
    def get_farmer_profile(self, farmer_id):
        """Get farmer profile information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        farmer_id, username, email, phone, full_name, farm_name,
                        district, village, farm_area, crop_types, registration_date,
                        last_login, profile_picture
                    FROM farmers WHERE farmer_id = ?
                """, (farmer_id,))
                
                farmer_data = cursor.fetchone()
                
                if farmer_data:
                    return {
                        "success": True,
                        "profile": {
                            "farmer_id": farmer_data[0],
                            "username": farmer_data[1],
                            "email": farmer_data[2],
                            "phone": farmer_data[3],
                            "full_name": farmer_data[4],
                            "farm_name": farmer_data[5],
                            "district": farmer_data[6],
                            "village": farmer_data[7],
                            "farm_area": farmer_data[8],
                            "crop_types": farmer_data[9],
                            "registration_date": farmer_data[10],
                            "last_login": farmer_data[11],
                            "profile_picture": farmer_data[12]
                        }
                    }
                else:
                    return {"success": False, "message": "Farmer profile not found"}
                    
        except Exception as e:
            return {"success": False, "message": f"Profile retrieval failed: {str(e)}"}

# Initialize the database when imported
if __name__ == "__main__":
    # Test the database initialization
    auth_db = FarmerAuthDB()
    print("Database initialized successfully!")