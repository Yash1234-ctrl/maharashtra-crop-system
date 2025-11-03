#!/usr/bin/env python3
"""
Simple Weather Location Service
Uses Google Geocoding API (basic) + Visual Crossing Weather API
"""

import requests
import argparse

class SimpleWeatherLocationService:
    def __init__(self, weather_api_key, google_api_key):
        self.weather_api_key = weather_api_key
        self.google_api_key = google_api_key
        self.weather_base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        self.geocode_base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    def geocode_location(self, address):
        """Convert address to coordinates using Google Geocoding API"""
        params = {
            'address': address,
            'key': self.google_api_key
        }
        
        try:
            response = requests.get(self.geocode_base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'OK' and data['results']:
                result = data['results'][0]
                location = result['geometry']['location']
                return {
                    'success': True,
                    'lat': location['lat'],
                    'lng': location['lng'],
                    'formatted_address': result['formatted_address']
                }
            else:
                return {
                    'success': False,
                    'error': f"Geocoding failed: {data.get('error_message', data['status'])}"
                }
                
        except requests.exceptions.RequestException as e:
            return {'success': False, 'error': f"Error geocoding: {e}"}
    
    def get_weather(self, lat, lng, location_name, forecast_days=0):
        """Get weather data for coordinates"""
        try:
            if forecast_days == 0:
                url = f"{self.weather_base_url}/{lat},{lng}/today"
                params = {
                    'unitGroup': 'metric',
                    'key': self.weather_api_key,
                    'include': 'current'
                }
            else:
                url = f"{self.weather_base_url}/{lat},{lng}"
                params = {
                    'unitGroup': 'metric',
                    'key': self.weather_api_key,
                    'include': 'days',
                    'elements': 'datetime,tempmax,tempmin,temp,humidity,precip,windspeed,conditions,description'
                }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            weather_data = response.json()
            
            return {
                'success': True,
                'location': location_name,
                'coordinates': {'lat': lat, 'lng': lng},
                'weather_data': weather_data
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Error fetching weather: {e}",
                'location': location_name,
                'coordinates': {'lat': lat, 'lng': lng}
            }
    
    def search_weather(self, location_query, forecast_days=0):
        """Search for location and get weather"""
        print(f"🔍 Looking up location: {location_query}")
        
        geocode_result = self.geocode_location(location_query)
        if not geocode_result['success']:
            return geocode_result
        
        print(f"📍 Found: {geocode_result['formatted_address']}")
        print(f"🗺️  Coordinates: {geocode_result['lat']:.4f}, {geocode_result['lng']:.4f}")
        
        return self.get_weather(
            geocode_result['lat'],
            geocode_result['lng'],
            geocode_result['formatted_address'],
            forecast_days
        )

def print_weather_result(result, show_forecast=False):
    """Print weather information"""
    if not result['success']:
        print(f"❌ {result.get('error', 'Unknown error')}")
        return
    
    weather_data = result['weather_data']
    location = result['location']
    coords = result['coordinates']
    
    print(f"\n🌤️  Weather for {location}")
    print("=" * 60)
    
    # Current weather
    if 'currentConditions' in weather_data:
        current = weather_data['currentConditions']
        print(f"🌡️  Temperature: {current.get('temp', 'N/A')}°C (feels like {current.get('feelslike', 'N/A')}°C)")
        print(f"☁️  Conditions: {current.get('conditions', 'N/A')}")
        print(f"💧 Humidity: {current.get('humidity', 'N/A')}%")
        print(f"💨 Wind: {current.get('windspeed', 'N/A')} km/h")
        print(f"👁️  Visibility: {current.get('visibility', 'N/A')} km")
        print(f"📊 Pressure: {current.get('pressure', 'N/A')} mb")
    
    # Forecast
    if show_forecast and 'days' in weather_data:
        print(f"\n📅 5-Day Forecast:")
        print("-" * 40)
        
        for i, day in enumerate(weather_data['days'][:5]):
            day_name = "Today" if i == 0 else day['datetime']
            print(f"\n{day_name}:")
            print(f"  🌡️  High: {day.get('tempmax', 'N/A')}°C | Low: {day.get('tempmin', 'N/A')}°C")
            print(f"  ☁️  {day.get('conditions', 'N/A')}")
            print(f"  💧 {day.get('humidity', 'N/A')}% humidity | 🌧️  {day.get('precip', 'N/A')}mm rain")
            print(f"  📝 {day.get('description', 'N/A')}")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Simple Weather Location Service')
    parser.add_argument('location', help='Location to search for')
    parser.add_argument('--forecast', '-f', action='store_true', help='Include 5-day forecast')
    parser.add_argument('--weather-key', default='5T4RN5AM9XUUVEEHMCJNCDCC4',
                       help='Visual Crossing Weather API key')
    parser.add_argument('--google-key', default='AIzaSyAOVYRIgupAurZup5y1PRh8Ismb1A3lLao',
                       help='Google Maps API key')
    
    args = parser.parse_args()
    
    service = SimpleWeatherLocationService(args.weather_key, args.google_key)
    
    forecast_days = 5 if args.forecast else 0
    result = service.search_weather(args.location, forecast_days)
    
    print_weather_result(result, show_forecast=args.forecast)

if __name__ == "__main__":
    main()