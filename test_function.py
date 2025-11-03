import requests
import json

# Test data
test_data = {
    "nitrogen": 90,
    "phosphorus": 42,
    "potassium": 43,
    "temperature": 25,
    "humidity": 82,
    "ph": 6.5,
    "rainfall": 200
}

def test_crop_prediction():
    # Replace this URL with your actual Firebase Function URL
    url = "https://us-central1-agro-dashboard.cloudfunctions.net/predict_crop"
    
    try:
        # Make the POST request
        response = requests.post(url, json=test_data)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            print("\nSuccess! Here's what the model predicted:")
            print("----------------------------------------")
            print(f"Predicted Crop: {result.get('predicted_crop')}")
            print(f"Confidence: {result.get('confidence')}%")
            print("\nTop Predictions:")
            for pred in result.get('top_predictions', []):
                print(f"- {pred['crop']}: {pred['probability']:.2f}%")
        else:
            print(f"\nError: Request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")
    except json.JSONDecodeError:
        print("\nError: Could not parse response as JSON")

if __name__ == "__main__":
    print("Testing Crop Prediction Function...")
    print(f"Sending test data: {json.dumps(test_data, indent=2)}")
    test_crop_prediction()