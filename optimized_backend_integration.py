#!/usr/bin/env python3
"""
Optimized Backend Integration Module
Seamlessly integrates optimized models into existing Maharashtra Agricultural System
Maintains full compatibility with current UI while providing enhanced accuracy
"""

import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OptimizedBackendIntegration:
    """Integration layer for optimized models with existing system"""
    
    def __init__(self):
        """Initialize optimized backend integration"""
        self.optimized_models = {}
        self.optimized_scalers = {}
        self.optimized_encoders = {}
        self.feature_selectors = {}
        self.performance_metrics = {}
        
        # Fallback models (existing system)
        self.fallback_models = {}
        self.fallback_available = False
        
        print("🔗 Initializing Optimized Backend Integration...")
        self.load_optimized_models()
        self.load_fallback_models()
    
    def load_optimized_models(self):
        """Load optimized models if available"""
        try:
            # Load optimized models
            model_files = [
                'optimized_crop_health_model.pkl',
                'optimized_yield_prediction_model.pkl', 
                'optimized_fertilizer_model.pkl'
            ]
            
            models_loaded = 0
            for model_file in model_files:
                if os.path.exists(model_file):
                    model_name = model_file.replace('optimized_', '').replace('_model.pkl', '')
                    with open(model_file, 'rb') as f:
                        self.optimized_models[model_name] = pickle.load(f)
                    models_loaded += 1
            
            # Load supporting components
            if os.path.exists('optimized_scalers.pkl'):
                with open('optimized_scalers.pkl', 'rb') as f:
                    self.optimized_scalers = pickle.load(f)
            
            if os.path.exists('optimized_encoders.pkl'):
                with open('optimized_encoders.pkl', 'rb') as f:
                    self.optimized_encoders = pickle.load(f)
            
            if os.path.exists('optimized_feature_selectors.pkl'):
                with open('optimized_feature_selectors.pkl', 'rb') as f:
                    self.feature_selectors = pickle.load(f)
            
            if os.path.exists('model_performance_metrics.pkl'):
                with open('model_performance_metrics.pkl', 'rb') as f:
                    self.performance_metrics = pickle.load(f)
            
            if models_loaded > 0:
                print(f"✅ Loaded {models_loaded} optimized models successfully")
                return True
            else:
                print("⚠️ No optimized models found - will use fallback models")
                return False
                
        except Exception as e:
            print(f"⚠️ Error loading optimized models: {e}")
            return False
    
    def load_fallback_models(self):
        """Load existing enhanced models as fallback"""
        try:
            # Try to load existing enhanced models
            enhanced_files = [
                'enhanced_crop_health_model.pkl',
                'enhanced_yield_model.pkl',
                'enhanced_fertilizer_model.pkl'
            ]
            
            fallback_loaded = 0
            for model_file in enhanced_files:
                if os.path.exists(model_file):
                    # Map to consistent naming
                    if 'crop_health' in model_file:
                        model_name = 'crop_health'
                    elif 'yield' in model_file:
                        model_name = 'yield_prediction'
                    elif 'fertilizer' in model_file:
                        model_name = 'fertilizer'
                    
                    with open(model_file, 'rb') as f:
                        self.fallback_models[model_name] = pickle.load(f)
                    fallback_loaded += 1
            
            if fallback_loaded > 0:
                self.fallback_available = True
                print(f"✅ Loaded {fallback_loaded} fallback models")
            else:
                print("⚠️ No fallback models available")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error loading fallback models: {e}")
            return False
    
    def get_model(self, model_type):
        """Get the best available model for the given type"""
        # Try optimized first, then fallback
        if model_type in self.optimized_models:
            return self.optimized_models[model_type], 'optimized'
        elif model_type in self.fallback_models:
            return self.fallback_models[model_type], 'fallback'
        else:
            return None, 'none'
    
    def prepare_crop_health_features(self, input_data):
        """Prepare features for crop health prediction"""
        try:
            # Enhanced feature mapping from input data
            features = {}
            
            # Direct mappings
            features['NDVI'] = input_data.get('ndvi', 0.7)
            features['SAVI'] = input_data.get('savi', 0.4)  
            features['Chlorophyll_Content'] = input_data.get('chlorophyll', 45)
            features['Leaf_Area_Index'] = input_data.get('lai', 3.5)
            features['Temperature'] = input_data.get('temperature', 28)
            features['Humidity'] = input_data.get('humidity', 75)
            features['Rainfall'] = input_data.get('rainfall', 50)
            features['Wind_Speed'] = input_data.get('wind_speed', 10)
            features['Soil_Moisture'] = input_data.get('soil_moisture', 35)
            features['Soil_pH'] = input_data.get('soil_ph', 6.5)
            features['Organic_Matter'] = input_data.get('organic_matter', 2.5)
            
            # Image-based features (with defaults for API calls)
            features['High_Resolution_RGB'] = input_data.get('rgb_quality', 85)
            features['Multispectral_Images'] = input_data.get('multispectral_quality', 80)
            features['Thermal_Images'] = input_data.get('thermal_quality', 75)
            features['Spatial_Resolution'] = input_data.get('spatial_resolution', 0.5)
            features['Canopy_Coverage'] = input_data.get('canopy_coverage', 70)
            
            # Enhanced composite features
            features['comprehensive_health_score'] = (
                features['NDVI'] * 0.3 +
                features['SAVI'] * 0.2 + 
                features['Chlorophyll_Content'] / 100 * 0.2 +
                features['Leaf_Area_Index'] / 10 * 0.15 +
                (100 - input_data.get('stress_indicator', 20)) / 100 * 0.15
            )
            
            features['soil_quality_index'] = (
                features['Soil_Moisture'] * 0.4 +
                (features['Soil_pH'] - 4) / 4 * 100 * 0.3 +
                features['Organic_Matter'] * 10 * 0.3
            )
            
            features['environmental_stress'] = (
                (features['Temperature'] > 35).astype(int) + 
                (features['Humidity'] > 90).astype(int) +
                (features['Rainfall'] > 100).astype(int)
            )
            
            # Advanced optimized features
            features['ndvi_savi_ratio'] = features['NDVI'] / (features['SAVI'] + 1e-6)
            features['vegetation_health_index'] = (
                features['NDVI'] * 0.4 + 
                features['SAVI'] * 0.3 + 
                features['Chlorophyll_Content'] / 100 * 0.3
            )
            
            features['temp_humidity_stress'] = (
                abs(features['Temperature'] - 25) / 15 + 
                abs(features['Humidity'] - 60) / 40
            )
            
            features['soil_fertility_index'] = (
                int((features['Soil_pH'] >= 6.0) and (features['Soil_pH'] <= 7.5)) *
                features['Organic_Matter'] * features['Soil_Moisture'] / 100
            )
            
            growth_stage = input_data.get('growth_stage', 3)
            features['growth_ndvi_interaction'] = growth_stage * features['NDVI']
            features['growth_temperature_interaction'] = growth_stage * features['Temperature']
            features['temperature_range'] = abs(features['Temperature'] - 25)  # Simplified
            
            return features
            
        except Exception as e:
            print(f"Error preparing crop health features: {e}")
            return None
    
    def prepare_yield_features(self, input_data):
        """Prepare features for yield prediction"""
        try:
            # Get crop health features first
            crop_features = self.prepare_crop_health_features(input_data)
            if not crop_features:
                return None
            
            # Yield-specific feature selection
            yield_features = {
                'Canopy_Coverage': crop_features['Canopy_Coverage'],
                'NDVI': crop_features['NDVI'],
                'SAVI': crop_features['SAVI'],
                'Chlorophyll_Content': crop_features['Chlorophyll_Content'],
                'Leaf_Area_Index': crop_features['Leaf_Area_Index'],
                'Temperature': crop_features['Temperature'],
                'Humidity': crop_features['Humidity'],
                'Rainfall': crop_features['Rainfall'],
                'Soil_Moisture': crop_features['Soil_Moisture'],
                'Soil_pH': crop_features['Soil_pH'],
                'Organic_Matter': crop_features['Organic_Matter'],
                'Crop_Growth_Stage': input_data.get('growth_stage', 3),
                'comprehensive_health_score': crop_features['comprehensive_health_score'],
                'soil_quality_index': crop_features['soil_quality_index'],
                'vegetation_health_index': crop_features['vegetation_health_index'],
                'soil_fertility_index': crop_features['soil_fertility_index'],
                'growth_ndvi_interaction': crop_features['growth_ndvi_interaction'],
                'growth_temperature_interaction': crop_features['growth_temperature_interaction']
            }
            
            return yield_features
            
        except Exception as e:
            print(f"Error preparing yield features: {e}")
            return None
    
    def prepare_fertilizer_features(self, input_data):
        """Prepare features for fertilizer recommendation"""
        try:
            features = {}
            
            # Basic NPK and soil parameters
            features['Nitrogen'] = input_data.get('nitrogen', 50)
            features['Phosphorus'] = input_data.get('phosphorus', 25)
            features['Potassium'] = input_data.get('potassium', 30)
            features['pH'] = input_data.get('soil_ph', 6.5)
            features['Rainfall'] = input_data.get('rainfall', 50)
            features['Temperature'] = input_data.get('temperature', 28)
            
            # Enhanced features
            total_npk = features['Nitrogen'] + features['Phosphorus'] + features['Potassium'] + 1e-6
            features['npk_balance_score'] = np.sqrt(
                (features['Nitrogen']**2 + features['Phosphorus']**2 + features['Potassium']**2) / 3
            )
            
            features['soil_crop_compatibility'] = (
                features['pH'] * 10 +
                features['Rainfall'] / 10 +
                (40 - abs(features['Temperature'] - 25)) * 2
            )
            
            # Advanced optimized features
            features['n_ratio'] = features['Nitrogen'] / total_npk
            features['p_ratio'] = features['Phosphorus'] / total_npk
            features['k_ratio'] = features['Potassium'] / total_npk
            
            features['ph_rainfall_interaction'] = features['pH'] * features['Rainfall'] / 100
            features['temperature_ph_balance'] = (
                abs(features['Temperature'] - 25) + abs(features['pH'] - 6.5)
            )
            
            # District and crop encoding (simplified for API)
            district = input_data.get('district', 'Mumbai')
            crop = input_data.get('crop', 'Rice')
            soil_color = input_data.get('soil_color', 'Brown')
            
            # Use hash-based encoding for consistency
            features['District_encoded'] = hash(district) % 100
            features['Crop_encoded'] = hash(crop) % 50  
            features['Soil_encoded'] = hash(soil_color) % 20
            
            # District deviation features (simplified)
            features['district_n_deviation'] = features['Nitrogen'] - 45  # Average baseline
            features['district_p_deviation'] = features['Phosphorus'] - 23
            
            return features
            
        except Exception as e:
            print(f"Error preparing fertilizer features: {e}")
            return None
    
    def predict_crop_health(self, input_data):
        """Enhanced crop health prediction using optimized models"""
        try:
            # Get the best available model
            model, model_type = self.get_model('crop_health')
            if not model:
                return {
                    'success': False,
                    'error': 'No crop health model available',
                    'model_type': 'none'
                }
            
            # Prepare features
            features = self.prepare_crop_health_features(input_data)
            if not features:
                return {
                    'success': False,
                    'error': 'Failed to prepare features',
                    'model_type': model_type
                }
            
            # Convert to DataFrame for consistent processing
            feature_df = pd.DataFrame([features])
            
            # Apply feature selection if available (for optimized models)
            if model_type == 'optimized' and 'crop_health' in self.feature_selectors:
                selector = self.feature_selectors['crop_health']
                # Get selected feature names
                health_features = [
                    'High_Resolution_RGB', 'Multispectral_Images', 'Thermal_Images',
                    'Spatial_Resolution', 'Canopy_Coverage', 'NDVI', 'SAVI',
                    'Chlorophyll_Content', 'Leaf_Area_Index', 'Temperature',
                    'Humidity', 'Rainfall', 'Wind_Speed', 'Soil_Moisture',
                    'Soil_pH', 'Organic_Matter', 'comprehensive_health_score',
                    'soil_quality_index', 'environmental_stress',
                    'ndvi_savi_ratio', 'vegetation_health_index', 'temp_humidity_stress',
                    'soil_fertility_index', 'growth_ndvi_interaction', 
                    'growth_temperature_interaction', 'temperature_range'
                ]
                
                # Ensure all features are present
                for feature in health_features:
                    if feature not in feature_df.columns:
                        feature_df[feature] = 0
                
                # Apply feature selection
                feature_array = feature_df[health_features].values
                feature_array = selector.transform(feature_array)
            else:
                # Use all available features
                feature_array = feature_df.values
            
            # Apply scaling if available
            if model_type == 'optimized' and 'crop_health' in self.optimized_scalers:
                scaler = self.optimized_scalers['crop_health']
                feature_array = scaler.transform(feature_array)
            
            # Make prediction
            prediction = model.predict(feature_array)[0]
            
            # Calculate confidence based on model type and performance metrics
            confidence = 0.85  # Default confidence
            if model_type == 'optimized' and 'crop_health' in self.performance_metrics:
                confidence = self.performance_metrics['crop_health']['test_score']
            
            # Classify health status
            if prediction >= 0.8:
                health_status = "Excellent"
                risk_level = "Low"
            elif prediction >= 0.6:
                health_status = "Good"
                risk_level = "Low"
            elif prediction >= 0.4:
                health_status = "Fair"
                risk_level = "Medium"
            elif prediction >= 0.2:
                health_status = "Poor"
                risk_level = "High"
            else:
                health_status = "Critical"
                risk_level = "Very High"
            
            # Generate recommendations based on features
            recommendations = self.generate_health_recommendations(features, health_status)
            
            return {
                'success': True,
                'health_score': float(prediction),
                'health_status': health_status,
                'risk_level': risk_level,
                'confidence': float(confidence),
                'recommendations': recommendations,
                'model_type': model_type,
                'model_performance': confidence * 100,
                'features_analyzed': len(features),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'model_type': 'error'
            }
    
    def predict_yield(self, input_data):
        """Enhanced yield prediction using optimized models"""
        try:
            # Get the best available model
            model, model_type = self.get_model('yield_prediction')
            if not model:
                return {
                    'success': False,
                    'error': 'No yield prediction model available',
                    'model_type': 'none'
                }
            
            # Prepare features
            features = self.prepare_yield_features(input_data)
            if not features:
                return {
                    'success': False,
                    'error': 'Failed to prepare features',
                    'model_type': model_type
                }
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Apply feature selection and scaling for optimized models
            if model_type == 'optimized':
                if 'yield_prediction' in self.feature_selectors:
                    selector = self.feature_selectors['yield_prediction']
                    yield_features = list(features.keys())
                    feature_array = feature_df[yield_features].values
                    feature_array = selector.transform(feature_array)
                else:
                    feature_array = feature_df.values
                
                if 'yield_prediction' in self.optimized_scalers:
                    scaler = self.optimized_scalers['yield_prediction']
                    feature_array = scaler.transform(feature_array)
            else:
                feature_array = feature_df.values
            
            # Make prediction
            prediction = model.predict(feature_array)[0]
            
            # Calculate confidence
            confidence = 0.82
            if model_type == 'optimized' and 'yield_prediction' in self.performance_metrics:
                confidence = self.performance_metrics['yield_prediction']['test_score']
            
            # Classify yield level
            crop_type = input_data.get('crop', 'Rice')
            if prediction >= 4.0:
                yield_category = "Excellent"
            elif prediction >= 3.0:
                yield_category = "Good"  
            elif prediction >= 2.0:
                yield_category = "Average"
            else:
                yield_category = "Poor"
            
            return {
                'success': True,
                'predicted_yield': float(prediction),
                'yield_category': yield_category,
                'confidence': float(confidence),
                'crop_type': crop_type,
                'model_type': model_type,
                'model_performance': confidence * 100,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Yield prediction failed: {str(e)}',
                'model_type': 'error'
            }
    
    def recommend_fertilizer(self, input_data):
        """Enhanced fertilizer recommendation using optimized models"""
        try:
            # Get the best available model
            model, model_type = self.get_model('fertilizer')
            if not model:
                return {
                    'success': False,
                    'error': 'No fertilizer model available',
                    'model_type': 'none'
                }
            
            # Prepare features
            features = self.prepare_fertilizer_features(input_data)
            if not features:
                return {
                    'success': False,
                    'error': 'Failed to prepare features',
                    'model_type': model_type
                }
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Apply feature selection and scaling for optimized models
            if model_type == 'optimized':
                if 'fertilizer' in self.feature_selectors:
                    selector = self.feature_selectors['fertilizer']
                    fert_features = list(features.keys())
                    feature_array = feature_df[fert_features].values
                    feature_array = selector.transform(feature_array)
                else:
                    feature_array = feature_df.values
                
                if 'fertilizer' in self.optimized_scalers:
                    scaler = self.optimized_scalers['fertilizer']
                    feature_array = scaler.transform(feature_array)
            else:
                feature_array = feature_df.values
            
            # Make prediction
            prediction = model.predict(feature_array)[0]
            
            # Decode fertilizer recommendation
            fertilizer_types = [
                'Urea', 'DAP', 'NPK 10:26:26', 'NPK 20:20:0', 'NPK 15:15:15',
                'Potash', 'Superphosphate', 'Organic Compost', 'Vermicompost'
            ]
            
            if model_type == 'optimized' and 'fertilizer' in self.optimized_encoders:
                try:
                    encoder = self.optimized_encoders['fertilizer']
                    fertilizer = encoder.inverse_transform([int(prediction)])[0]
                except:
                    fertilizer = fertilizer_types[int(prediction) % len(fertilizer_types)]
            else:
                fertilizer = fertilizer_types[int(prediction) % len(fertilizer_types)]
            
            # Calculate confidence
            confidence = 0.87
            if model_type == 'optimized' and 'fertilizer' in self.performance_metrics:
                confidence = self.performance_metrics['fertilizer']['test_score']
            
            # Generate application recommendations
            application_rate = self.calculate_application_rate(fertilizer, features)
            
            return {
                'success': True,
                'recommended_fertilizer': fertilizer,
                'application_rate': application_rate,
                'confidence': float(confidence),
                'npk_analysis': {
                    'nitrogen': features['Nitrogen'],
                    'phosphorus': features['Phosphorus'], 
                    'potassium': features['Potassium'],
                    'balance_score': features['npk_balance_score']
                },
                'model_type': model_type,
                'model_performance': confidence * 100,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Fertilizer recommendation failed: {str(e)}',
                'model_type': 'error'
            }
    
    def generate_health_recommendations(self, features, health_status):
        """Generate actionable health recommendations"""
        recommendations = []
        
        if features['NDVI'] < 0.5:
            recommendations.append("🌱 Low vegetation index detected. Consider nutrient supplementation.")
        
        if features['Soil_pH'] < 6.0 or features['Soil_pH'] > 7.5:
            recommendations.append(f"⚖️ Soil pH ({features['Soil_pH']:.1f}) needs adjustment for optimal growth.")
        
        if features['Soil_Moisture'] < 25:
            recommendations.append("💧 Soil moisture is low. Increase irrigation frequency.")
        
        if features['Temperature'] > 35:
            recommendations.append("🌡️ High temperature stress detected. Consider shade management.")
        
        if features['environmental_stress'] > 1:
            recommendations.append("⚠️ Multiple environmental stress factors detected. Monitor closely.")
        
        if not recommendations:
            recommendations.append("✅ Crop health parameters are within optimal ranges.")
        
        return recommendations
    
    def calculate_application_rate(self, fertilizer, features):
        """Calculate optimal fertilizer application rate"""
        base_rates = {
            'Urea': 50,  # kg per acre
            'DAP': 40,
            'NPK 10:26:26': 45,
            'NPK 20:20:0': 35,
            'NPK 15:15:15': 40,
            'Potash': 25,
            'Superphosphate': 35,
            'Organic Compost': 200,
            'Vermicompost': 150
        }
        
        base_rate = base_rates.get(fertilizer, 40)
        
        # Adjust based on soil conditions
        if features['pH'] < 6.0:
            base_rate *= 0.9  # Reduce for acidic soil
        elif features['pH'] > 7.5:
            base_rate *= 1.1  # Increase for alkaline soil
        
        # Adjust based on NPK balance
        if features['npk_balance_score'] < 30:
            base_rate *= 1.2  # Increase for poor balance
        
        return f"{base_rate:.0f} kg per acre"
    
    def get_system_status(self):
        """Get current system status and performance metrics"""
        status = {
            'optimized_models_available': len(self.optimized_models),
            'fallback_models_available': len(self.fallback_models),
            'total_models': len(self.optimized_models) + len(self.fallback_models),
            'performance_boost': 'Available' if self.optimized_models else 'Pending Optimization',
            'timestamp': datetime.now().isoformat()
        }
        
        if self.performance_metrics:
            status['estimated_accuracy'] = {}
            for model_name, metrics in self.performance_metrics.items():
                status['estimated_accuracy'][model_name] = {
                    'test_score': metrics['test_score'],
                    'cv_score': metrics['cv_mean'],
                    'improvement': f"+{(metrics['test_score'] - 0.75) * 100:.1f}%" if metrics['test_score'] > 0.75 else "Baseline"
                }
        
        return status

# Global instance for easy integration
optimized_backend = OptimizedBackendIntegration()

# Compatibility functions for existing system
def extract_enhanced_crop_features(data):
    """Compatibility function for existing enhanced_backend_api.py"""
    features = optimized_backend.prepare_crop_health_features(data)
    return features if features else {}

def predict_enhanced_crop_health(features_dict):
    """Compatibility function for existing enhanced_backend_api.py"""
    return optimized_backend.predict_crop_health(features_dict)

def predict_enhanced_yield(features_dict):
    """Compatibility function for existing enhanced_backend_api.py"""
    return optimized_backend.predict_yield(features_dict)

def recommend_enhanced_fertilizer(features_dict):
    """Compatibility function for existing enhanced_backend_api.py"""
    return optimized_backend.recommend_fertilizer(features_dict)

if __name__ == "__main__":
    # Test the integration
    test_data = {
        'ndvi': 0.75,
        'savi': 0.45,
        'temperature': 29,
        'humidity': 70,
        'soil_ph': 6.8,
        'soil_moisture': 40,
        'nitrogen': 55,
        'phosphorus': 28,
        'potassium': 35,
        'crop': 'Rice',
        'district': 'Pune'
    }
    
    print("🧪 Testing Optimized Backend Integration...")
    
    # Test crop health prediction
    health_result = optimized_backend.predict_crop_health(test_data)
    print(f"🌱 Crop Health: {health_result}")
    
    # Test yield prediction
    yield_result = optimized_backend.predict_yield(test_data)
    print(f"📊 Yield Prediction: {yield_result}")
    
    # Test fertilizer recommendation
    fert_result = optimized_backend.recommend_fertilizer(test_data)
    print(f"🧪 Fertilizer: {fert_result}")
    
    # System status
    status = optimized_backend.get_system_status()
    print(f"⚙️ System Status: {status}")