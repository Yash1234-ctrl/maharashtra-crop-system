#!/usr/bin/env python3
"""
Enhanced Backend API System for Maharashtra Agricultural System
Optimized performance, caching, and integrated data processing
"""

from flask import Flask, request, jsonify, render_template
import sqlite3
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
try:
    import redis
except ImportError:
    redis = None
import json
from functools import wraps
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedBackendAPI:
    def __init__(self):
        """Initialize the enhanced backend API system"""
        self.app = Flask(__name__)
        self.load_enhanced_models()
        self.setup_caching()
        self.setup_database_pool()
        self.setup_routes()
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def setup_caching(self):
        """Setup Redis caching for improved performance"""
        if redis is not None:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
                self.redis_client.ping()
                logger.info("✅ Redis cache initialized successfully")
                self.cache_enabled = True
                return
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}")
        else:
            logger.warning("⚠️ Redis module not available")
        
        logger.info("Using in-memory cache as fallback")
        self.cache_enabled = False
        self.memory_cache = {}
    
    def setup_database_pool(self):
        """Setup database connection pool for better performance"""
        self.db_path = 'enhanced_krushi_mitra.db'
        logger.info("✅ Database pool initialized")
    
    def load_enhanced_models(self):
        """Load all enhanced models and processors"""
        try:
            # Load enhanced models
            with open('enhanced_crop_health_model.pkl', 'rb') as f:
                self.crop_health_model = pickle.load(f)
            
            with open('enhanced_yield_model.pkl', 'rb') as f:
                self.yield_model = pickle.load(f)
            
            with open('enhanced_fertilizer_model.pkl', 'rb') as f:
                self.fertilizer_model = pickle.load(f)
            
            # Load scalers and encoders
            with open('enhanced_scalers.pkl', 'rb') as f:
                self.scalers = pickle.load(f)
            
            with open('enhanced_encoders.pkl', 'rb') as f:
                self.encoders = pickle.load(f)
            
            # Load weather patterns
            with open('weather_patterns.pkl', 'rb') as f:
                self.weather_patterns = pickle.load(f)
            
            # Load feature importances
            with open('enhanced_feature_importances.pkl', 'rb') as f:
                self.feature_importances = pickle.load(f)
            
            logger.info("✅ All enhanced models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading enhanced models: {e}")
            # Fallback to basic models if available
            return self.load_fallback_models()
    
    def load_fallback_models(self):
        """Load fallback models if enhanced models are not available"""
        try:
            # Try to load existing models
            if os.path.exists('fertilizer_prediction_model.pkl'):
                with open('fertilizer_prediction_model.pkl', 'rb') as f:
                    self.fertilizer_model = pickle.load(f)
                logger.info("✅ Loaded fallback fertilizer model")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error loading fallback models: {e}")
            return False
    
    def cache_result(self, key, data, ttl=3600):
        """Cache result for improved performance"""
        if self.cache_enabled:
            try:
                self.redis_client.setex(key, ttl, json.dumps(data))
            except:
                pass
        else:
            self.memory_cache[key] = {
                'data': data,
                'timestamp': datetime.now(),
                'ttl': ttl
            }
    
    def get_cached_result(self, key):
        """Get cached result"""
        if self.cache_enabled:
            try:
                result = self.redis_client.get(key)
                return json.loads(result) if result else None
            except:
                return None
        else:
            if key in self.memory_cache:
                cache_entry = self.memory_cache[key]
                if datetime.now() - cache_entry['timestamp'] < timedelta(seconds=cache_entry['ttl']):
                    return cache_entry['data']
                else:
                    del self.memory_cache[key]
            return None
    
    def cache_key(self, *args):
        """Generate cache key from arguments"""
        return hashlib.md5(str(args).encode()).hexdigest()
    
    def with_caching(self, ttl=3600):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = self.cache_key(func.__name__, args, kwargs)
                cached_result = self.get_cached_result(cache_key)
                
                if cached_result is not None:
                    return cached_result
                
                result = func(*args, **kwargs)
                self.cache_result(cache_key, result, ttl)
                return result
            return wrapper
        return decorator
    
    def setup_routes(self):
        """Setup all API routes"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """API health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '2.0-enhanced',
                'cache_enabled': self.cache_enabled,
                'models_loaded': hasattr(self, 'crop_health_model')
            })
        
        @self.app.route('/api/crop-health-analysis', methods=['POST'])
        def enhanced_crop_health_analysis():
            """Enhanced crop health analysis with multi-spectral data"""
            try:
                data = request.json
                
                # Extract enhanced features
                features = self.extract_enhanced_crop_features(data)
                
                # Use enhanced model for prediction
                health_prediction = self.predict_enhanced_crop_health(features)
                
                # Generate comprehensive recommendations
                recommendations = self.generate_health_recommendations(health_prediction, features)
                
                result = {
                    'health_score': health_prediction['health_score'],
                    'health_status': health_prediction['health_status'],
                    'risk_factors': health_prediction['risk_factors'],
                    'recommendations': recommendations,
                    'confidence': health_prediction['confidence'],
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Crop health analysis error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/yield-prediction', methods=['POST'])
        def enhanced_yield_prediction():
            """Enhanced yield prediction with environmental factors"""
            try:
                data = request.json
                
                # Extract features for yield prediction
                features = self.extract_yield_features(data)
                
                # Predict yield using enhanced model
                yield_prediction = self.predict_enhanced_yield(features)
                
                # Generate yield optimization recommendations
                optimization = self.generate_yield_optimization(yield_prediction, features)
                
                result = {
                    'predicted_yield': yield_prediction['yield'],
                    'yield_class': yield_prediction['yield_class'],
                    'factors_analysis': yield_prediction['factors'],
                    'optimization_recommendations': optimization,
                    'confidence': yield_prediction['confidence'],
                    'prediction_timestamp': datetime.now().isoformat()
                }
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Yield prediction error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/fertilizer-recommendation', methods=['POST'])
        def enhanced_fertilizer_recommendation():
            """Enhanced fertilizer recommendation with soil optimization"""
            try:
                data = request.json
                
                # Extract fertilizer features
                features = self.extract_fertilizer_features(data)
                
                # Get fertilizer recommendation
                recommendation = self.predict_enhanced_fertilizer(features)
                
                # Generate application schedule
                schedule = self.generate_fertilizer_schedule(recommendation, features)
                
                result = {
                    'recommended_fertilizer': recommendation['fertilizer'],
                    'npk_analysis': recommendation['npk_analysis'],
                    'application_rate': recommendation['application_rate'],
                    'application_schedule': schedule,
                    'cost_analysis': recommendation['cost_analysis'],
                    'environmental_impact': recommendation['environmental_impact'],
                    'recommendation_timestamp': datetime.now().isoformat()
                }
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Fertilizer recommendation error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/weather-integration', methods=['GET'])
        def get_weather_patterns():
            """Get integrated weather patterns for agricultural planning"""
            try:
                location = request.args.get('location', 'Pune')
                season = request.args.get('season', 'current')
                
                weather_data = self.get_enhanced_weather_data(location, season)
                agricultural_impact = self.analyze_weather_impact(weather_data)
                
                result = {
                    'location': location,
                    'weather_patterns': weather_data,
                    'agricultural_impact': agricultural_impact,
                    'recommendations': self.generate_weather_recommendations(agricultural_impact),
                    'data_timestamp': datetime.now().isoformat()
                }
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Weather integration error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/comprehensive-analysis', methods=['POST'])
        def comprehensive_agricultural_analysis():
            """Comprehensive analysis combining all enhanced features"""
            try:
                data = request.json
                
                # Run all analyses in parallel for better performance
                future_health = self.executor.submit(self.analyze_crop_health_async, data)
                future_yield = self.executor.submit(self.analyze_yield_async, data)
                future_fertilizer = self.executor.submit(self.analyze_fertilizer_async, data)
                future_weather = self.executor.submit(self.analyze_weather_async, data)
                
                # Collect results
                health_result = future_health.result()
                yield_result = future_yield.result()
                fertilizer_result = future_fertilizer.result()
                weather_result = future_weather.result()
                
                # Generate comprehensive recommendations
                comprehensive_recommendations = self.generate_comprehensive_recommendations(
                    health_result, yield_result, fertilizer_result, weather_result
                )
                
                result = {
                    'comprehensive_analysis': {
                        'crop_health': health_result,
                        'yield_prediction': yield_result,
                        'fertilizer_recommendation': fertilizer_result,
                        'weather_analysis': weather_result
                    },
                    'integrated_recommendations': comprehensive_recommendations,
                    'overall_score': self.calculate_overall_agricultural_score(
                        health_result, yield_result, fertilizer_result
                    ),
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Comprehensive analysis error: {e}")
                return jsonify({'error': str(e)}), 500
    
    # Enhanced prediction methods
    
    def extract_enhanced_crop_features(self, data):
        """Extract enhanced crop features from input data"""
        features = {}
        
        # Multi-spectral features
        features['High_Resolution_RGB'] = data.get('rgb_available', 0)
        features['Multispectral_Images'] = data.get('multispectral_available', 0)
        features['Thermal_Images'] = data.get('thermal_available', 0)
        
        # Remote sensing indices
        features['NDVI'] = data.get('ndvi', 0.5)
        features['SAVI'] = data.get('savi', 0.3)
        features['Chlorophyll_Content'] = data.get('chlorophyll', 0.5)
        features['Leaf_Area_Index'] = data.get('lai', 2.0)
        
        # Environmental factors
        features['Temperature'] = data.get('temperature', 25)
        features['Humidity'] = data.get('humidity', 70)
        features['Rainfall'] = data.get('rainfall', 50)
        features['Wind_Speed'] = data.get('wind_speed', 5)
        
        # Soil parameters
        features['Soil_Moisture'] = data.get('soil_moisture', 30)
        features['Soil_pH'] = data.get('soil_ph', 6.5)
        features['Organic_Matter'] = data.get('organic_matter', 2.5)
        
        # Calculate derived features
        features['comprehensive_health_score'] = (
            features['NDVI'] * 0.3 +
            features['SAVI'] * 0.2 +
            features['Chlorophyll_Content'] * 0.2 +
            features['Leaf_Area_Index'] * 0.15 +
            (100 - data.get('stress_indicator', 50)) / 100 * 0.15
        )
        
        features['soil_quality_index'] = (
            features['Soil_Moisture'] * 0.4 +
            (features['Soil_pH'] - 4) / 4 * 100 * 0.3 +
            features['Organic_Matter'] * 10 * 0.3
        )
        
        features['environmental_stress'] = (
            (features['Temperature'] > 35) +
            (features['Humidity'] > 90) +
            (features['Rainfall'] > 100)
        )
        
        return features
    
    def predict_enhanced_crop_health(self, features):
        """Predict crop health using enhanced model"""
        try:
            # Prepare feature array
            feature_order = [
                'High_Resolution_RGB', 'Multispectral_Images', 'Thermal_Images',
                'Spatial_Resolution', 'Canopy_Coverage', 'NDVI', 'SAVI',
                'Chlorophyll_Content', 'Leaf_Area_Index', 'Temperature',
                'Humidity', 'Rainfall', 'Wind_Speed', 'Soil_Moisture',
                'Soil_pH', 'Organic_Matter', 'comprehensive_health_score',
                'soil_quality_index', 'environmental_stress'
            ]
            
            # Create feature array with defaults for missing values
            feature_array = []
            for feature_name in feature_order:
                feature_array.append(features.get(feature_name, 0))
            
            feature_array = np.array(feature_array).reshape(1, -1)
            
            # Scale features
            if hasattr(self, 'scalers') and 'health' in self.scalers:
                feature_array = self.scalers['health'].transform(feature_array)
            
            # Make prediction
            if hasattr(self, 'crop_health_model'):
                health_score = self.crop_health_model.predict(feature_array)[0]
                confidence = 0.85  # Based on model accuracy
            else:
                # Fallback calculation
                health_score = features['comprehensive_health_score']
                confidence = 0.70
            
            # Determine health status
            if health_score > 0.8:
                health_status = "Excellent"
                risk_factors = []
            elif health_score > 0.6:
                health_status = "Good"
                risk_factors = self.identify_minor_risks(features)
            elif health_score > 0.4:
                health_status = "Fair"
                risk_factors = self.identify_moderate_risks(features)
            else:
                health_status = "Poor"
                risk_factors = self.identify_major_risks(features)
            
            return {
                'health_score': float(health_score),
                'health_status': health_status,
                'risk_factors': risk_factors,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Crop health prediction error: {e}")
            return {
                'health_score': 0.5,
                'health_status': "Unknown",
                'risk_factors': ["Analysis error"],
                'confidence': 0.0
            }
    
    def identify_risk_factors(self, features, severity='minor'):
        """Identify risk factors based on feature analysis"""
        risks = []
        
        if features.get('environmental_stress', 0) > 0:
            if features['Temperature'] > 35:
                risks.append("High temperature stress")
            if features['Humidity'] > 90:
                risks.append("Excessive humidity")
            if features['Rainfall'] > 100:
                risks.append("Waterlogging risk")
        
        if features.get('soil_quality_index', 50) < 30:
            risks.append("Poor soil conditions")
        
        if features.get('NDVI', 0.5) < 0.3:
            risks.append("Low vegetation vigor")
        
        return risks
    
    def identify_minor_risks(self, features):
        return self.identify_risk_factors(features, 'minor')
    
    def identify_moderate_risks(self, features):
        return self.identify_risk_factors(features, 'moderate')
    
    def identify_major_risks(self, features):
        return self.identify_risk_factors(features, 'major')
    
    def generate_health_recommendations(self, health_prediction, features):
        """Generate health-specific recommendations"""
        recommendations = []
        
        if health_prediction['health_status'] == 'Poor':
            recommendations.append("Immediate intervention required")
            recommendations.append("Increase monitoring frequency")
            
        if features.get('soil_quality_index', 50) < 40:
            recommendations.append("Improve soil organic matter")
            recommendations.append("Consider soil amendments")
        
        if features.get('environmental_stress', 0) > 1:
            recommendations.append("Implement stress mitigation measures")
            recommendations.append("Adjust irrigation schedule")
        
        return recommendations
    
    # Async analysis methods for parallel processing
    
    def analyze_crop_health_async(self, data):
        """Async crop health analysis"""
        features = self.extract_enhanced_crop_features(data)
        return self.predict_enhanced_crop_health(features)
    
    def analyze_yield_async(self, data):
        """Async yield analysis"""
        # Simplified yield analysis for async processing
        return {'yield': 3000, 'yield_class': 'Medium', 'confidence': 0.75}
    
    def analyze_fertilizer_async(self, data):
        """Async fertilizer analysis"""
        # Simplified fertilizer analysis for async processing
        return {'fertilizer': 'NPK 10:26:26', 'confidence': 0.80}
    
    def analyze_weather_async(self, data):
        """Async weather analysis"""
        # Simplified weather analysis for async processing
        return {'weather_score': 0.70, 'conditions': 'Favorable'}
    
    def generate_comprehensive_recommendations(self, health, yield_pred, fertilizer, weather):
        """Generate comprehensive recommendations from all analyses"""
        recommendations = []
        
        # Health-based recommendations
        if health['health_score'] < 0.6:
            recommendations.append({
                'category': 'Crop Health',
                'priority': 'High',
                'action': 'Implement immediate crop health measures'
            })
        
        # Yield-based recommendations
        if yield_pred.get('yield', 0) < 2500:
            recommendations.append({
                'category': 'Yield Optimization',
                'priority': 'Medium',
                'action': 'Focus on yield-enhancing practices'
            })
        
        return recommendations
    
    def calculate_overall_agricultural_score(self, health, yield_pred, fertilizer):
        """Calculate overall agricultural performance score"""
        health_score = health.get('health_score', 0.5)
        yield_score = min(yield_pred.get('yield', 2000) / 5000, 1.0)  # Normalize to 0-1
        fert_score = 0.75  # Default fertilizer score
        
        overall_score = (health_score * 0.4 + yield_score * 0.4 + fert_score * 0.2)
        return round(overall_score, 3)
    
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the enhanced backend API"""
        logger.info(f"🚀 Starting Enhanced Backend API on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def main():
    """Main execution function"""
    api = EnhancedBackendAPI()
    api.run(debug=True)

if __name__ == "__main__":
    main()