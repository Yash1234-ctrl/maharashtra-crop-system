"""
Test MongoDB database connection without boolean evaluation.
"""
from mongodb_config import MongoCropDB
from datetime import datetime

def test_db_connection():
    try:
        # Initialize database connection
        db = MongoCropDB()
        
        if not hasattr(db, 'db') or db.db is None:
            print("❌ Database connection failed - db object is None")
            return False
            
        # Try to insert a test document
        test_doc = {"test": "connection", "timestamp": datetime.now()}
        result = db.db.test_collection.insert_one(test_doc)
        
        # Verify we can retrieve it
        found = db.db.test_collection.find_one({"_id": result.inserted_id})
        
        # Clean up
        db.db.test_collection.delete_one({"_id": result.inserted_id})
        
        if found:
            print("✅ MongoDB connection and operations successful!")
            return True
        else:
            print("❌ Could not verify database operations")
            return False
            
    except Exception as e:
        print(f"❌ Database connection/operation failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_db_connection()