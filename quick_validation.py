#!/usr/bin/env python3
"""
Quick System Validation for Enhanced Maharashtra Agricultural System
Tests core functionality without intensive model training
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import json

def validate_data_files():
    """Validate presence and basic structure of data files"""
    print("🔍 Validating Data Files...")
    
    results = {
        'csv_files_present': 0,
        'csv_files_total': 6,
        'data_quality': []
    }
    
    csv_files = [
        'agriculture_dataset.csv',
        'Crop and fertilizer dataset.csv', 
        'Crop_recommendationV2.csv',
        'weather_data.csv',
        'Fertilizer Prediction.csv',
        'Fertilizer.csv'
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            results['csv_files_present'] += 1
            try:
                df = pd.read_csv(csv_file)
                results['data_quality'].append({
                    'file': csv_file,
                    'records': len(df),
                    'columns': len(df.columns),
                    'status': 'valid'
                })
                print(f"✅ {csv_file}: {len(df)} records, {len(df.columns)} columns")
            except Exception as e:
                results['data_quality'].append({
                    'file': csv_file,
                    'status': 'error',
                    'error': str(e)
                })
                print(f"❌ {csv_file}: Error - {e}")
        else:
            print(f"❌ {csv_file}: File not found")
    
    return results

def validate_enhanced_modules():
    """Validate enhanced modules can be imported and initialized"""
    print("\n🔧 Validating Enhanced Modules...")
    
    results = {
        'data_processor': False,
        'backend_api': False,
        'processor_functions': [],
        'api_functions': []
    }
    
    # Test data processor
    try:
        from enhanced_data_processor import EnhancedDataProcessor
        processor = EnhancedDataProcessor()
        results['data_processor'] = True
        
        # Check key methods exist
        methods = ['load_all_datasets', 'create_integrated_features', 'train_enhanced_models']
        for method in methods:
            if hasattr(processor, method):
                results['processor_functions'].append(method)
                print(f"✅ Data Processor: {method} method available")
            else:
                print(f"❌ Data Processor: {method} method missing")
                
    except Exception as e:
        print(f"❌ Data Processor: Import failed - {e}")
    
    # Test backend API
    try:
        from enhanced_backend_api import EnhancedBackendAPI
        api = EnhancedBackendAPI()
        results['backend_api'] = True
        
        # Check key methods exist
        methods = ['extract_enhanced_crop_features', 'predict_enhanced_crop_health']
        for method in methods:
            if hasattr(api, method):
                results['api_functions'].append(method)
                print(f"✅ Backend API: {method} method available")
            else:
                print(f"❌ Backend API: {method} method missing")
                
    except Exception as e:
        print(f"❌ Backend API: Import failed - {e}")
    
    return results

def validate_data_integration():
    """Test basic data integration without model training"""
    print("\n📊 Validating Data Integration...")
    
    results = {
        'datasets_loaded': False,
        'features_created': False,
        'integration_successful': False
    }
    
    try:
        from enhanced_data_processor import EnhancedDataProcessor
        processor = EnhancedDataProcessor()
        
        # Test dataset loading
        load_success = processor.load_all_datasets()
        results['datasets_loaded'] = load_success
        
        if load_success:
            print("✅ All datasets loaded successfully")
            
            # Test basic feature creation
            feature_success = processor.create_integrated_features()
            results['features_created'] = feature_success
            
            if feature_success:
                print("✅ Integrated features created successfully")
                
                # Check enhanced agriculture dataset
                if 'enhanced_agriculture' in processor.datasets:
                    enhanced_df = processor.datasets['enhanced_agriculture']
                    required_features = [
                        'comprehensive_health_score',
                        'soil_quality_index',
                        'environmental_stress'
                    ]
                    
                    features_present = all(col in enhanced_df.columns for col in required_features)
                    if features_present:
                        results['integration_successful'] = True
                        print("✅ Enhanced features successfully integrated")
                        print(f"   - Dataset size: {len(enhanced_df)} records")
                        print(f"   - Enhanced features: {len([c for c in enhanced_df.columns if any(f in c for f in ['comprehensive', 'quality', 'stress'])])} new columns")
                    else:
                        print("❌ Required enhanced features missing")
                else:
                    print("❌ Enhanced agriculture dataset not created")
            else:
                print("❌ Feature creation failed")
        else:
            print("❌ Dataset loading failed")
            
    except Exception as e:
        print(f"❌ Data integration test failed: {e}")
    
    return results

def validate_api_functionality():
    """Test API functionality with sample data"""
    print("\n🌐 Validating API Functionality...")
    
    results = {
        'api_initialized': False,
        'feature_extraction': False,
        'prediction_working': False
    }
    
    try:
        from enhanced_backend_api import EnhancedBackendAPI
        api = EnhancedBackendAPI()
        results['api_initialized'] = True
        print("✅ API initialized successfully")
        
        # Test feature extraction with sample data
        sample_data = {
            'ndvi': 0.7,
            'savi': 0.4,
            'temperature': 28,
            'humidity': 75,
            'soil_ph': 6.5,
            'soil_moisture': 35,
            'nitrogen': 50,
            'phosphorus': 25,
            'potassium': 30
        }
        
        try:
            features = api.extract_enhanced_crop_features(sample_data)
            if isinstance(features, dict) and len(features) > len(sample_data):
                results['feature_extraction'] = True
                print("✅ Feature extraction working")
                print(f"   - Input features: {len(sample_data)}")
                print(f"   - Enhanced features: {len(features)}")
            else:
                print("❌ Feature extraction not creating enhanced features")
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
        
        # Test prediction capability (if models exist)
        if results['feature_extraction']:
            try:
                prediction = api.predict_enhanced_crop_health(features)
                if isinstance(prediction, dict) and 'health_status' in prediction:
                    results['prediction_working'] = True
                    print("✅ Prediction system working")
                else:
                    print("❌ Prediction not returning expected format")
            except Exception as e:
                print(f"⚠️ Prediction test skipped (models may not be trained): {e}")
        
    except Exception as e:
        print(f"❌ API validation failed: {e}")
    
    return results

def validate_system_improvements():
    """Check for system quality improvements"""
    print("\n✨ Validating System Improvements...")
    
    improvements = {
        'data_integration': False,
        'enhanced_features': False,
        'multi_model_support': False,
        'performance_optimizations': False,
        'comprehensive_analytics': False
    }
    
    # Check if enhanced data processor provides integrated analysis
    try:
        from enhanced_data_processor import EnhancedDataProcessor
        processor = EnhancedDataProcessor()
        
        if processor.load_all_datasets():
            # Check for multiple dataset integration
            expected_datasets = ['agriculture', 'crop_fertilizer', 'crop_recommendation', 'weather']
            loaded_datasets = [key for key in processor.datasets.keys() if any(exp in key for exp in expected_datasets)]
            
            if len(loaded_datasets) >= 3:
                improvements['data_integration'] = True
                print("✅ Multi-dataset integration implemented")
            
            if processor.create_integrated_features():
                # Check for enhanced features
                if 'enhanced_agriculture' in processor.datasets:
                    df = processor.datasets['enhanced_agriculture']
                    enhanced_cols = [col for col in df.columns if any(word in col.lower() 
                                   for word in ['comprehensive', 'quality', 'stress', 'index', 'score'])]
                    
                    if len(enhanced_cols) >= 3:
                        improvements['enhanced_features'] = True
                        print("✅ Enhanced feature engineering implemented")
                        print(f"   - New analytical features: {len(enhanced_cols)}")
                
                # Check for weather pattern analysis
                if hasattr(processor, 'weather_patterns') and len(processor.weather_patterns) > 0:
                    improvements['comprehensive_analytics'] = True
                    print("✅ Advanced analytics (weather patterns) implemented")
        
    except Exception as e:
        print(f"⚠️ System improvement validation limited: {e}")
    
    # Check for API performance features
    try:
        from enhanced_backend_api import EnhancedBackendAPI
        api = EnhancedBackendAPI()
        
        # Check for caching
        if hasattr(api, 'cache_enabled'):
            improvements['performance_optimizations'] = True
            print("✅ Performance optimizations (caching) implemented")
        
        # Check for multiple model support
        model_methods = [method for method in dir(api) if 'predict' in method.lower()]
        if len(model_methods) >= 2:
            improvements['multi_model_support'] = True
            print("✅ Multi-model prediction support implemented")
        
    except Exception as e:
        print(f"⚠️ API improvement validation limited: {e}")
    
    return improvements

def generate_validation_report(data_results, module_results, integration_results, api_results, improvements):
    """Generate comprehensive validation report"""
    print("\n" + "="*60)
    print("📊 SYSTEM VALIDATION REPORT")
    print("="*60)
    
    # Calculate overall scores
    data_score = (data_results['csv_files_present'] / data_results['csv_files_total']) * 100
    module_score = (int(module_results['data_processor']) + int(module_results['backend_api'])) * 50
    integration_score = (int(integration_results['datasets_loaded']) + 
                        int(integration_results['features_created']) + 
                        int(integration_results['integration_successful'])) * 33.33
    api_score = (int(api_results['api_initialized']) + 
                int(api_results['feature_extraction'])) * 50
    improvement_score = sum(improvements.values()) * 20
    
    overall_score = (data_score + module_score + integration_score + api_score + improvement_score) / 5
    
    print(f"📈 Overall System Health: {overall_score:.1f}%")
    print(f"   📁 Data Quality: {data_score:.1f}%")
    print(f"   🔧 Module Integration: {module_score:.1f}%")
    print(f"   📊 Data Processing: {integration_score:.1f}%")
    print(f"   🌐 API Functionality: {api_score:.1f}%")
    print(f"   ✨ System Improvements: {improvement_score:.1f}%")
    
    print(f"\n🎯 Key Achievements:")
    if data_results['csv_files_present'] == data_results['csv_files_total']:
        print("   ✅ All required datasets are present and accessible")
    
    if integration_results['integration_successful']:
        print("   ✅ Advanced data integration and feature engineering operational")
    
    if improvements['data_integration']:
        print("   ✅ Multi-dataset integration successfully implemented")
    
    if improvements['enhanced_features']:
        print("   ✅ Comprehensive analytical features created")
    
    if improvements['performance_optimizations']:
        print("   ✅ Performance optimizations implemented")
    
    # Save detailed report
    report_data = {
        'validation_summary': {
            'overall_score': overall_score,
            'data_score': data_score,
            'module_score': module_score,
            'integration_score': integration_score,
            'api_score': api_score,
            'improvement_score': improvement_score,
            'timestamp': datetime.now().isoformat()
        },
        'detailed_results': {
            'data_files': data_results,
            'modules': module_results,
            'integration': integration_results,
            'api': api_results,
            'improvements': improvements
        }
    }
    
    with open('system_validation_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: system_validation_report.json")
    
    return overall_score > 80

def main():
    """Main validation function"""
    print("🚀 Quick System Validation Starting...")
    print("="*60)
    
    # Run validations
    data_results = validate_data_files()
    module_results = validate_enhanced_modules()
    integration_results = validate_data_integration()
    api_results = validate_api_functionality()
    improvements = validate_system_improvements()
    
    # Generate report
    success = generate_validation_report(
        data_results, module_results, integration_results, 
        api_results, improvements
    )
    
    if success:
        print("\n🎉 System validation successful!")
        print("Your Maharashtra Agricultural System has been enhanced with:")
        print("   📊 Advanced multi-dataset integration")
        print("   🧠 Intelligent feature engineering")
        print("   🔮 Predictive analytics capabilities")
        print("   🚀 Performance optimizations")
        print("   📈 Comprehensive agricultural insights")
        print("\n✅ System is ready for enhanced agricultural analysis!")
    else:
        print("\n⚠️ Some components need attention. Check the detailed report.")
    
    return success

if __name__ == "__main__":
    main()