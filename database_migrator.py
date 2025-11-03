import sqlite3
from firebase_config import FirebaseManager, db
from datetime import datetime

class DatabaseMigrator:
    def __init__(self, sqlite_db_path="farmer_auth.db"):
        self.sqlite_db_path = sqlite_db_path
        self.firebase = FirebaseManager()
    
    async def migrate_users(self):
        """Migrate users from SQLite to Firebase"""
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT farmer_id, username, email, phone, full_name, 
                           farm_name, district, village, farm_area, crop_types,
                           registration_date, last_login, profile_picture
                    FROM farmers
                """)
                users = cursor.fetchall()
                
                batch = db.batch()
                migrated_count = 0
                
                for user in users:
                    user_data = {
                        'farmer_id': user[0],
                        'username': user[1],
                        'email': user[2],
                        'phone': user[3],
                        'full_name': user[4],
                        'farm_name': user[5],
                        'district': user[6],
                        'village': user[7],
                        'farm_area': user[8],
                        'crop_types': user[9].split(',') if user[9] else [],
                        'registration_date': user[10],
                        'last_login': user[11],
                        'profile_picture': user[12],
                        'migrated_at': datetime.now(),
                        'role': 'farmer'
                    }
                    
                    # Create a new document in the users collection
                    user_ref = db.collection('users').document()
                    batch.set(user_ref, user_data)
                    migrated_count += 1
                    
                    # Commit batch after every 500 users
                    if migrated_count % 500 == 0:
                        batch.commit()
                        batch = db.batch()
                
                # Commit any remaining users
                if migrated_count % 500 != 0:
                    batch.commit()
                
                return {
                    'success': True,
                    'message': f'Successfully migrated {migrated_count} users to Firebase'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Migration failed: {str(e)}'
            }
    
    async def migrate_preferences(self):
        """Migrate user preferences to Firebase"""
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT farmer_id, language, theme, notifications_enabled,
                           email_alerts, sms_alerts, weather_alerts, pest_alerts
                    FROM farmer_preferences
                """)
                preferences = cursor.fetchall()
                
                batch = db.batch()
                migrated_count = 0
                
                for pref in preferences:
                    pref_data = {
                        'farmer_id': pref[0],
                        'language': pref[1],
                        'theme': pref[2],
                        'notifications': {
                            'enabled': bool(pref[3]),
                            'email': bool(pref[4]),
                            'sms': bool(pref[5]),
                            'weather': bool(pref[6]),
                            'pest': bool(pref[7])
                        },
                        'updated_at': datetime.now()
                    }
                    
                    # Store preferences as a subcollection of users
                    pref_ref = db.collection('users').document(str(pref[0]))\
                                .collection('preferences').document('settings')
                    batch.set(pref_ref, pref_data)
                    migrated_count += 1
                    
                    if migrated_count % 500 == 0:
                        batch.commit()
                        batch = db.batch()
                
                if migrated_count % 500 != 0:
                    batch.commit()
                
                return {
                    'success': True,
                    'message': f'Successfully migrated {migrated_count} preferences to Firebase'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Preferences migration failed: {str(e)}'
            }
    
    async def migrate_all(self):
        """Migrate all data to Firebase"""
        results = {
            'users': await self.migrate_users(),
            'preferences': await self.migrate_preferences()
        }
        return results