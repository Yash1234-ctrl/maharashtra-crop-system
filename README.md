# 🌱 Plant Disease Detection AI

An advanced AI-powered web application for detecting plant diseases and stress using deep learning. This application can identify various plant diseases and provide health assessments with confidence scores and actionable recommendations.

![Plant Disease Detection AI](https://img.shields.io/badge/AI-Plant%20Disease%20Detection-green)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.3.3-red.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.13.0-orange.svg)

## ✨ Features

- **🤖 Advanced AI Detection**: Deep learning model trained on plant disease datasets
- **📊 Probability Analysis**: Detailed confidence scores for all detected conditions
- **🎨 Modern UI**: Beautiful, responsive web interface with plant-themed design
- **📱 Mobile Friendly**: Works perfectly on desktop and mobile devices
- **⚡ Real-time Results**: Instant disease detection and analysis
- **💡 Smart Recommendations**: Actionable advice based on detected conditions
- **🔒 Secure Upload**: Safe file handling with validation and cleanup

## 🚀 Currently Supported Detections

- **Potato Early Blight** - Fungal disease affecting potato plants
- **Tomato Late Blight** - Serious disease affecting tomato plants
- **Healthy Plants** - Detection of healthy tomato plants

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 2: Verify Model Files

Make sure you have the following files in your project directory:
- `best_model.h5` - Your trained model
- `class_names.txt` - Class labels for the model

### Step 3: Run the Application

```bash
# Start the web application
python web_app.py
```

The application will be available at: **http://localhost:5000**

## 📖 How to Use

1. **Open the Application**: Navigate to `http://localhost:5000` in your web browser
2. **Upload Image**: 
   - Drag and drop a plant image onto the upload area, OR
   - Click "Choose Image" to browse and select a file
3. **Wait for Analysis**: The AI will process your image (usually takes 2-5 seconds)
4. **View Results**: 
   - See the disease detection results
   - Check confidence scores and health assessment
   - Read personalized recommendations
5. **Try Another Image**: Click "Analyze Another Image" to test more plants

## 🎯 Supported Image Formats

- **JPEG/JPG** - Most common format
- **PNG** - High quality images
- **GIF** - Animated or static images
- **BMP** - Bitmap images
- **TIFF** - High resolution images
- **WebP** - Modern web format

**Maximum file size**: 16MB

## 📊 Understanding Results

### Health Score
- **90-100%**: Excellent plant health
- **70-89%**: Good health with minor concerns
- **50-69%**: Moderate health issues
- **Below 50%**: Significant health problems

### Disease Probability
- Shows the likelihood of disease presence
- Higher percentages indicate more concerning conditions
- Healthy plants will show low disease probability

### Recommendations
The AI provides tailored advice such as:
- Treatment suggestions for detected diseases
- Prevention measures
- Monitoring recommendations
- When to consult agricultural experts

## 🔧 Project Structure

```
GC2/
├── web_app.py              # Main Flask application
├── templates/
│   └── index.html          # Web interface template
├── uploads/                # Temporary upload directory
├── dataset/                # Training data (if available)
├── best_model.h5          # Trained AI model
├── class_names.txt        # Model class labels
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🌐 API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Image upload and prediction endpoint
- `GET /health` - API health check

## 🔬 Model Information

- **Architecture**: Transfer Learning with MobileNetV2
- **Input Size**: 224x224 pixels
- **Training Data**: Plant disease datasets with multiple crop types
- **Accuracy**: High accuracy for supported plant diseases

## 🚀 Deployment Options

### Local Development
```bash
python web_app.py
```

### Production with Gunicorn (Linux/Mac)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 web_app:app
```

### Production with Waitress (Windows)
```bash
pip install waitress
waitress-serve --host 0.0.0.0 --port 5000 web_app:app
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**Model Loading Error**
- Ensure `best_model.h5` exists in the project directory
- Check that TensorFlow is properly installed

**Upload Errors**
- Verify image file is in supported format
- Check file size is under 16MB
- Ensure stable internet connection

**Performance Issues**
- Close unnecessary applications
- Use smaller image files
- Consider upgrading hardware for faster processing

## 📞 Support

If you encounter any issues or have questions:
1. Check this README for solutions
2. Review error messages in the browser console
3. Ensure all dependencies are properly installed

## 🌟 Future Enhancements

- [ ] Support for more crop types and diseases
- [ ] Batch processing for multiple images
- [ ] Historical analysis tracking
- [ ] Integration with weather data
- [ ] Mobile app version
- [ ] API for third-party integrations

---

**Made with ❤️ for sustainable agriculture**

## 📦 Large Files (Hosted Externally)

Due to GitHub’s 100 MB file limit, large assets are hosted externally on Google Drive.

| File | Description | Download Link |
|------|--------------|----------------|
| `agriculture_dataset.csv` | Main agricultural dataset for Maharashtra | [Download](https://drive.google.com/uc?export=download&id=1gYrPlQFe9vJUA2lefT4t16u2Odka32Ze) |
| `fertilizer_prediction_model.pkl` | Trained fertilizer recommendation model | [Download](https://drive.google.com/uc?export=download&id=1ZNYqalYl1uATPF2g_bf9bgBcKtm2rbg8) |
| `maharashtra_agricultural_system_deployment.zip` | Deployment bundle | [Download](https://drive.google.com/uc?export=download&id=10cOBdfiss9IFIKf9P90GvswUxz6ARBth) |

If your code needs these files at runtime (e.g., Streamlit app or model loading), add the helper below near the top of your main Python file (example `maharashtra_crop_system.py`):

```python
import os
import requests

def download_file(url, filename):
   if not os.path.exists(filename):
      print(f"Downloading {filename} ...")
      response = requests.get(url)
      with open(filename, "wb") as f:
         f.write(response.content)
      print(f"{filename} downloaded successfully.")
   else:
      print(f"{filename} already exists.")

# Automatically download files if missing
download_file(
   "https://drive.google.com/uc?export=download&id=1gYrPlQFe9vJUA2lefT4t16u2Odka32Ze",
   "maharashtra_agri_deployment/data/agriculture_dataset.csv",
)

download_file(
   "https://drive.google.com/uc?export=download&id=1ZNYqalYl1uATPF2g_bf9bgBcKtm2rbg8",
   "maharashtra_agri_deployment/models/fertilizer_prediction_model.pkl",
)
```

Note: I updated `maharashtra_crop_system.py` to prefer the downloaded model under `maharashtra_agri_deployment/models/` and to keep a backward-compatible copy in the project root (`fertilizer_prediction_model.pkl`). Make sure `requests` is installed in your environment before running the downloader.