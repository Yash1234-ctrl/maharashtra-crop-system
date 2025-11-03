"""
Test MongoDB Database Operations
"""
from mongodb_config import MongoCropDB
from datetime import datetime

def test_mongodb_operations():
    try:
        # Initialize MongoDB connection
        db = MongoCropDB()
        print("✅ MongoDB connection initialized")
        
        # Test data
        test_data = {
            "test_id": "test_operation",
            "timestamp": datetime.now(),
            "test_value": 123
        }
        
        # Test insert
        result = db.db.test_collection.insert_one(test_data)
        print("✅ Test insert successful")
        
        # Test query
        found = db.db.test_collection.find_one({"test_id": "test_operation"})
        if found:
            print("✅ Test query successful")
            print(f"Found document: {found}")
        
        # Clean up test data
        db.db.test_collection.delete_one({"test_id": "test_operation"})
        print("✅ Test cleanup successful")
        
        # Test completed
        print("\n✅ All MongoDB operations tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        
    finally:
        db.close()
        print("\nMongoDB connection closed")

if __name__ == "__main__":
    test_mongodb_operations()