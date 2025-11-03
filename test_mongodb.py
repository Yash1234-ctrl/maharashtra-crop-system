"""
Test MongoDB Connection and Basic Operations
"""
from mongodb_config import MongoCropDB
from datetime import datetime

def test_mongodb_connection():
    try:
        # Initialize MongoDB connection
        db = MongoCropDB()
        
        # Test data
        test_data = {
            "farmer_id": "test_farmer",
            "district": "Pune",
            "crop_type": "Cotton",
            "analysis_date": datetime.now(),
            "soil_ph": 6.5,
            "nitrogen": 120,
            "phosphorus": 45,
            "potassium": 80,
            "test_entry": True
        }
        
        # Test insert
        result = db.soil_analysis.insert_one(test_data)
        print("✅ Insert Test: Successful")
        
        # Test query
        found = db.soil_analysis.find_one({"_id": result.inserted_id})
        print("✅ Query Test: Successful")
        print(f"Found document: {found}")
        
        # Clean up test data
        db.soil_analysis.delete_one({"_id": result.inserted_id})
        print("✅ Cleanup Test: Successful")
        
        # Close connection
        db.close()
        print("✅ Connection Test: All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_mongodb_connection()