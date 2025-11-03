"""
Simplified MongoDB Configuration for Maharashtra Crop System
Falls back to simulated data when MongoDB is not available
"""

class MongoCropDB:
    def __init__(self, connection_string=None):
        """Initialize without MongoDB - using simulated data"""
        self.using_fallback = True
        print("Using simulated data (MongoDB not connected)")
    
    def save_crop_analysis(self, analysis_data):
        """Simulate saving crop analysis"""
        return True
    
    def save_weather_data(self, weather_data):
        """Simulate saving weather data"""
        return True
    
    def save_pest_prediction(self, pest_data):
        """Simulate saving pest prediction"""
        return True
    
    def save_soil_analysis(self, soil_data):
        """Simulate saving soil analysis"""
        return True
    
    def get_farmer_history(self, farmer_id):
        """Return simulated farmer history"""
        return {
            'crop_analysis': [],
            'soil_analysis': []
        }
    
    def get_district_summary(self, district):
        """Return simulated district summary"""
        return []
    
    def get_weather_history(self, district, days=30):
        """Return simulated weather history"""
        return []
    
    def get_pest_alerts(self, district):
        """Return simulated pest alerts"""
        return []
    
    def close(self):
        """No connection to close in simulation"""
        pass