from pymongo import MongoClient

def test_mongo():
    try:
        # Create a MongoDB client
        client = MongoClient('mongodb+srv://yashimamdar_db_user:paulvrWJZqKz8SIJ@cluster0.r5cckg1.mongodb.net/maharashtra_agri_db?retryWrites=true&w=majority&appName=Cluster0')
        
        # Get the database
        db = client.maharashtra_agri_db
        
        # Try to ping the database
        client.admin.command('ping')
        
        print("✅ MongoDB connection successful!")
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    test_mongo()