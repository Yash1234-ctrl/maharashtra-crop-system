#!/usr/bin/env python3
"""
Maharashtra Agricultural System - Deployment Package Creator
Creates a deployable package with all necessary files for another device
"""

import os
import shutil
import zipfile
from pathlib import Path
import json

def create_deployment_package():
    """Create a complete deployment package"""
    
    # Define package structure
    package_files = {
        'essential': [
            'maharashtra_crop_system.py',
            'requirements.txt',
            '.env',
            '.env.example',
            'README.md',
            'DEPLOYMENT_GUIDE.md'
        ],
        'models': [
            'best_model.h5',
            'best_model_backup.h5',
            'class_names.txt',
            'enhanced_soil_model_features.pkl',
            'enhanced_soil_model_gradient_boosting.pkl',
            'enhanced_soil_model_label_encoder.pkl',
            'enhanced_soil_model_random_forest.pkl',
            'enhanced_soil_model_scalers.pkl',
            'fertilizer_prediction_model.pkl',
            'fertilizer_scaler.pkl'
        ],
        'databases': [
            'maharashtra_agri_system.db',
            'krushi_mitra.db',
            'enhanced_krushi_mitra.db'
        ],
        'data': [
            'agriculture_dataset.csv',
            'weather_data.csv',
            'pest_risk_dataset.csv',
            'Crop_recommendationV2.csv'
        ],
        'assets': [
            'agri_background.jpg'
        ]
    }
    
    # Create deployment directory
    deploy_dir = Path('maharashtra_agri_deployment')
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    # Copy files by category
    copied_files = []
    missing_files = []
    
    for category, files in package_files.items():
        category_dir = deploy_dir / category
        category_dir.mkdir(exist_ok=True)
        
        print(f"\n📁 Processing {category.upper()} files:")
        
        for file in files:
            if os.path.exists(file):
                try:
                    shutil.copy2(file, category_dir / file)
                    copied_files.append(f"{category}/{file}")
                    print(f"  ✅ {file}")
                except Exception as e:
                    print(f"  ❌ {file} - Error: {e}")
                    missing_files.append(f"{category}/{file} - {e}")
            else:
                print(f"  ⚠️ {file} - File not found")
                missing_files.append(f"{category}/{file} - Not found")
    
    # Copy directories
    directories_to_copy = [
        ('dataset', 'optional'),
        ('test_images', 'optional'),
        ('uploads', 'optional'),
        ('templates', 'optional')
    ]
    
    print(f"\n📁 Processing DIRECTORIES:")
    for dir_name, importance in directories_to_copy:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            try:
                shutil.copytree(dir_name, deploy_dir / dir_name)
                print(f"  ✅ {dir_name}/ - Directory copied")
                copied_files.append(f"directories/{dir_name}/")
            except Exception as e:
                print(f"  ❌ {dir_name}/ - Error: {e}")
                missing_files.append(f"directories/{dir_name}/ - {e}")
        else:
            status = "⚠️" if importance == "optional" else "❌"
            print(f"  {status} {dir_name}/ - Directory not found")
            if importance != "optional":
                missing_files.append(f"directories/{dir_name}/ - Not found")
    
    # Create deployment info
    deployment_info = {
        'package_created': str(Path.cwd()),
        'total_files_copied': len(copied_files),
        'copied_files': copied_files,
        'missing_files': missing_files,
        'python_version_required': '3.8+',
        'main_application': 'maharashtra_crop_system.py',
        'launch_command': 'streamlit run maharashtra_crop_system.py'
    }
    
    # Save deployment info
    with open(deploy_dir / 'deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    # Create quick start script
    quick_start_script = """#!/bin/bash
# Maharashtra Agricultural System - Quick Start Script

echo "🌾 Maharashtra Agricultural System - Quick Deployment"
echo "=================================================="

# Check Python version
python_version=$(python --version 2>&1 | grep -Po '(?<=Python )[0-9]+\\.[0-9]+')
echo "📍 Python version: $python_version"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create uploads directory if not exists
mkdir -p uploads

# Launch application
echo "🚀 Launching Maharashtra Agricultural System..."
echo "💡 Access your application at: http://localhost:8501"
streamlit run maharashtra_crop_system.py
"""
    
    with open(deploy_dir / 'quick_start.sh', 'w', encoding='utf-8') as f:
        f.write(quick_start_script)
    
    # Create Windows batch file
    windows_script = """@echo off
echo 🌾 Maharashtra Agricultural System - Quick Deployment
echo ==================================================

REM Check Python version
python --version

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Create uploads directory if not exists
if not exist "uploads" mkdir uploads

REM Launch application
echo 🚀 Launching Maharashtra Agricultural System...
echo 💡 Access your application at: http://localhost:8501
streamlit run maharashtra_crop_system.py

pause
"""
    
    with open(deploy_dir / 'quick_start.bat', 'w', encoding='utf-8') as f:
        f.write(windows_script)
    
    # Create README for deployment
    deployment_readme = f"""# Maharashtra Agricultural System - Deployment Package

## 📋 QUICK SETUP INSTRUCTIONS

### 1. System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 5GB free disk space
- Internet connection (optional)

### 2. Installation Steps

#### Windows:
1. Double-click `quick_start.bat`
2. Wait for dependencies to install
3. Application will launch automatically

#### Linux/macOS:
1. Make script executable: `chmod +x quick_start.sh`
2. Run script: `./quick_start.sh`
3. Application will launch automatically

#### Manual Installation:
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run maharashtra_crop_system.py
```

### 3. Access Application
- **Local URL**: http://localhost:8501
- **Network URL**: http://[your-ip]:8501

### 4. Package Contents
- **Essential Files**: {len([f for f in copied_files if f.startswith('essential')])} files
- **AI Models**: {len([f for f in copied_files if f.startswith('models')])} files
- **Databases**: {len([f for f in copied_files if f.startswith('databases')])} files
- **Data Files**: {len([f for f in copied_files if f.startswith('data')])} files
- **Total Files**: {len(copied_files)} files

### 5. Troubleshooting
- See `DEPLOYMENT_GUIDE.md` for detailed instructions
- Check `deployment_info.json` for package details
- Ensure all files from `essential/` folder are present

### 6. Support
If you encounter issues:
1. Verify Python 3.8+ is installed
2. Check internet connection for API calls
3. Ensure all essential files are present
4. Review the detailed deployment guide

**🎉 Your Maharashtra Agricultural System is ready to deploy!**
"""
    
    with open(deploy_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(deployment_readme)
    
    # Create ZIP package
    zip_filename = 'maharashtra_agricultural_system_deployment.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(deploy_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, deploy_dir)
                zipf.write(file_path, arc_path)
    
    # Print summary
    print(f"\n🎉 DEPLOYMENT PACKAGE CREATED SUCCESSFULLY!")
    print(f"📁 Package Directory: {deploy_dir}")
    print(f"📦 ZIP Package: {zip_filename}")
    print(f"✅ Files Copied: {len(copied_files)}")
    print(f"⚠️ Missing Files: {len(missing_files)}")
    
    if missing_files:
        print(f"\n⚠️ MISSING FILES (System will work without these):")
        for missing in missing_files[:10]:  # Show first 10
            print(f"  • {missing}")
        if len(missing_files) > 10:
            print(f"  • ... and {len(missing_files) - 10} more files")
    
    print(f"\n📋 DEPLOYMENT INSTRUCTIONS:")
    print(f"1. Copy '{zip_filename}' to target device")
    print(f"2. Extract the ZIP file")
    print(f"3. Run 'quick_start.bat' (Windows) or 'quick_start.sh' (Linux/macOS)")
    print(f"4. Access application at http://localhost:8501")
    
    return deploy_dir, zip_filename

if __name__ == "__main__":
    create_deployment_package()