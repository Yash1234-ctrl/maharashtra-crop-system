# 🌾 Maharashtra Agricultural System - Deployment Guide

## 📋 COMPLETE SETUP FOR NEW DEVICE

### 🔧 SYSTEM REQUIREMENTS
- **Python**: 3.8 or higher (Recommended: 3.9-3.12)
- **RAM**: Minimum 4GB (Recommended: 8GB+)
- **Storage**: 5GB free space
- **OS**: Windows, Linux, or macOS
- **Internet**: For API calls (optional - works offline too)

---

## 📦 STEP 1: INSTALL PYTHON DEPENDENCIES

### Method 1: Using requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```

### Method 2: Manual Installation
```bash
# Core Framework
pip install streamlit>=1.28.0

# Machine Learning & Image Processing
pip install tensorflow>=2.12.0
pip install Pillow>=10.0.0
pip install numpy>=1.21.0
pip install opencv-python-headless>=4.8.0

# Data Science
pip install pandas>=1.3.0
pip install scikit-learn>=1.0.0

# Visualization
pip install plotly>=5.15.0
pip install plotly-express>=0.4.0
pip install matplotlib>=3.5.0

# API & Web
pip install requests>=2.28.0

# Environment & Configuration
pip install python-dotenv>=1.0.0

# Additional Libraries
pip install python-dateutil>=2.8.0
pip install scipy>=1.10.0
```

---

## 📂 STEP 2: REQUIRED FILES TO COPY

### 🔴 ESSENTIAL FILES (Must Copy):
```
📁 Project Root/
├── maharashtra_crop_system.py          # Main application
├── requirements.txt                     # Dependencies list
├── .env                                # API keys configuration
├── .env.example                        # Environment template
└── README.md                           # Documentation
```

### 🟡 MODEL FILES (Copy if available):
```
📁 AI Models/
├── best_model.h5                       # TensorFlow disease detection model
├── best_model_backup.h5                # Backup model
├── class_names.txt                     # Disease class names
├── enhanced_soil_model_*.pkl           # Soil analysis models
├── fertilizer_prediction_model.pkl     # Fertilizer recommendation
└── fertilizer_scaler.pkl               # Data scaler
```

### 🟢 DATABASE FILES (Copy if needed):
```
📁 Databases/
├── maharashtra_agri_system.db          # Main system database
├── krushi_mitra.db                     # Legacy database
└── enhanced_krushi_mitra.db            # Enhanced features
```

### 🔵 DATASET (Copy if training/testing):
```
📁 dataset/
├── Potato___Early_blight/              # 779 images
├── Tomato_healthy/                     # 1,113 images
└── Tomato_Late_blight/                 # 779 images
```

### 🟣 OPTIONAL FILES:
```
📁 Supporting Files/
├── test_images/                        # Sample test images
├── uploads/                            # Upload directory
├── templates/                          # Custom templates
├── agriculture_dataset.csv             # Agricultural data
├── weather_data.csv                    # Weather patterns
├── pest_risk_dataset.csv              # Pest information
└── agri_background.jpg                 # Background image
```

---

## ⚙️ STEP 3: ENVIRONMENT CONFIGURATION

### Create `.env` file:
```env
# Maharashtra Agricultural System - API Configuration

# OpenWeatherMap API (Free: 1000 calls/day)
# Sign up: https://openweathermap.org/api
OPENWEATHER_API_KEY=6c46ecb55ba4725ad522158345263f56

# WeatherAPI (Free: 1M calls/month)
# Sign up: https://weatherapi.com/
WEATHERAPI_KEY=11409983432847d28e3181526251909

# Google Maps API (Optional - for geocoding)
# Sign up: https://developers.google.com/maps
GOOGLE_MAPS_API_KEY=your_google_api_key_here

# AgroMonitoring API (Free: 1000 calls/month)
# Sign up: https://agromonitoring.com/api
AGROMONITORING_API_KEY=0056bd905f981964d43b06ed7f31b31b

# Application Settings
FLASK_ENV=development
SECRET_KEY=maharashtra_krushi_mitra_2025_secure_key
DATABASE_URL=sqlite:///maharashtra_agri_system.db
APP_HOST=127.0.0.1
APP_PORT=8501
DEBUG=True
```

---

## 🚀 STEP 4: LAUNCH COMMANDS

### Windows:
```cmd
# Navigate to project directory
cd "C:\path\to\your\project"

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run maharashtra_crop_system.py
```

### Linux/macOS:
```bash
# Navigate to project directory
cd /path/to/your/project

# Install dependencies
pip3 install -r requirements.txt

# Run the application
streamlit run maharashtra_crop_system.py
```

### Using Virtual Environment (Recommended):
```bash
# Create virtual environment
python -m venv agri_env

# Activate environment
# Windows:
agri_env\Scripts\activate
# Linux/macOS:
source agri_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run maharashtra_crop_system.py
```

---

## 🌐 STEP 5: ACCESS URLS

After successful launch, access your application at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://[your-ip]:8501

---

## 🔧 TROUBLESHOOTING

### Common Issues & Solutions:

#### 1. TensorFlow Installation Issues:
```bash
# For older CPUs (without AVX)
pip install tensorflow-cpu==2.12.0

# For Apple Silicon Macs
pip install tensorflow-macos tensorflow-metal
```

#### 2. OpenCV Issues:
```bash
# Alternative OpenCV installation
pip uninstall opencv-python opencv-python-headless
pip install opencv-python-headless==4.8.1.78
```

#### 3. Streamlit Port Issues:
```bash
# Use different port
streamlit run maharashtra_crop_system.py --server.port 8502
```

#### 4. Memory Issues:
```bash
# Set TensorFlow memory limit
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
```

#### 5. API Key Issues:
- Ensure `.env` file is in the project root
- Check API key validity
- Verify internet connection

---

## 📱 DEPLOYMENT OPTIONS

### 1. Local Development:
- Follow steps above
- Access via localhost

### 2. Network Deployment:
```bash
streamlit run maharashtra_crop_system.py --server.address 0.0.0.0
```

### 3. Cloud Deployment:
- **Streamlit Cloud**: Connect GitHub repository
- **Heroku**: Use Procfile with web: streamlit run app.py
- **AWS/Azure**: Deploy using containers

### 4. Docker Deployment:
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "maharashtra_crop_system.py"]
```

---

## 📊 PERFORMANCE OPTIMIZATION

### For Better Performance:
1. **GPU Support**: Install `tensorflow-gpu` if NVIDIA GPU available
2. **Memory Management**: Set TensorFlow memory growth
3. **Caching**: Enable Streamlit caching for models
4. **Image Optimization**: Resize images before processing

### Minimum System Specs:
- **CPU**: 2 cores, 2.5GHz+
- **RAM**: 4GB (8GB recommended)
- **Storage**: 5GB free space
- **Network**: Broadband for API calls

---

## 🔐 SECURITY CONSIDERATIONS

1. **API Keys**: Never commit `.env` to version control
2. **Database**: Use proper database permissions
3. **File Uploads**: Validate file types and sizes
4. **Network**: Use HTTPS in production
5. **Updates**: Keep dependencies updated

---

## ✅ VERIFICATION CHECKLIST

- [ ] Python 3.8+ installed
- [ ] All dependencies from requirements.txt installed
- [ ] Main script `maharashtra_crop_system.py` present
- [ ] `.env` file configured with API keys
- [ ] Model files copied (if available)
- [ ] Application launches without errors
- [ ] Can access web interface
- [ ] APIs working (check weather data)
- [ ] Image upload functionality working
- [ ] Database operations successful

---

## 📞 SUPPORT

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all files are copied correctly
3. Ensure Python version compatibility
4. Check internet connection for API calls
5. Validate API keys in `.env` file

---

## 🔄 UPDATES

To update the system:
1. Pull latest code changes
2. Update requirements: `pip install -r requirements.txt --upgrade`
3. Restart the application
4. Clear browser cache if needed

---

**🎉 Your Maharashtra Agricultural System is now ready for deployment on any device!**