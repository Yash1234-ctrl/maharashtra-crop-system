#!/usr/bin/env python3
"""
Enhanced Weather Location Service
Combines Google Places API with Visual Crossing Weather API for comprehensive location-based weather data
"""

import requests
import json
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class GooglePlacesWeatherService:
    def __init__(self, weather_api_key: str, google_api_key: str):
        self.weather_api_key = weather_api_key
        self.google_api_key = google_api_key
        self.weather_base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        self.places_base_url = "https://maps.googleapis.com/maps/api"
    
    def search_places(self, query: str, location_type: str = None) -> List[Dict]:
        """Search for places using Google Places API"""
        url = f"{self.places_base_url}/place/textsearch/json"
        
        params = {
            'query': query,
            'key': self.google_api_key,
            'fields': 'name,formatted_address,geometry,place_id,types'
        }
        
        if location_type:
            params['type'] = location_type
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'OK':
                return data.get('results', [])[:5]  # Return top 5 results
            else:
                print(f"❌ Places API Error: {data.get('error_message', data['status'])}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error searching places: {e}")
            return []
    
    def get_place_details(self, place_id: str) -> Optional[Dict]:
        """Get detailed information about a specific place"""
        url = f"{self.places_base_url}/place/details/json"
        
        params = {
            'place_id': place_id,
            'key': self.google_api_key,
            'fields': 'name,formatted_address,geometry,place_id,types,vicinity,rating,user_ratings_total'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'OK':
                return data.get('result')
            else:
                print(f"❌ Place Details Error: {data.get('error_message', data['status'])}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting place details: {e}")
            return None
    
    def reverse_geocode(self, lat: float, lng: float) -> Optional[Dict]:
        """Convert coordinates to address information"""
        url = f"{self.places_base_url}/geocode/json"
        
        params = {
            'latlng': f"{lat},{lng}",
            'key': self.google_api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'OK' and data['results']:
                return data['results'][0]
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error reverse geocoding: {e}")
            return None
    
    def get_weather_for_place(self, place_info: Dict, forecast_days: int = 0) -> Dict:
        """Get weather data for a specific place"""
        if 'geometry' not in place_info:
            return {'success': False, 'error': 'No location data available'}
        
        location = place_info['geometry']['location']
        lat, lng = location['lat'], location['lng']
        place_name = place_info.get('formatted_address', place_info.get('name', 'Unknown'))
        
        return self.get_weather_by_coordinates(lat, lng, place_name, forecast_days)
    
    def get_weather_by_coordinates(self, lat: float, lng: float, location_name: str = None, forecast_days: int = 0) -> Dict:
        """Get weather data for specific coordinates"""
        if location_name is None:
            reverse_geocode_result = self.reverse_geocode(lat, lng)
            location_name = reverse_geocode_result.get('formatted_address', f'{lat:.4f}, {lng:.4f}') if reverse_geocode_result else f'{lat:.4f}, {lng:.4f}'
        
        try:
            if forecast_days == 0:
                # Current weather only
                url = f"{self.weather_base_url}/{lat},{lng}/today"
                params = {
                    'unitGroup': 'metric',
                    'key': self.weather_api_key,
                    'include': 'current'
                }
            else:
                # Forecast
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
                'weather_data': weather_data,
                'forecast_days': forecast_days
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Error fetching weather data: {e}",
                'location': location_name,
                'coordinates': {'lat': lat, 'lng': lng}
            }
    
    def search_and_get_weather(self, query: str, forecast_days: int = 0) -> List[Dict]:
        """Search for places and get weather data for each"""
        places = self.search_places(query)
        results = []
        
        for place in places:
            weather_result = self.get_weather_for_place(place, forecast_days)
            weather_result['place_info'] = place
            results.append(weather_result)
        
        return results

def print_place_weather(weather_result: Dict, show_forecast: bool = False):
    """Pretty print weather data for a place"""
    if not weather_result['success']:
        print(f"❌ {weather_result.get('error', 'Unknown error')}")
        return
    
    place_info = weather_result.get('place_info', {})
    weather_data = weather_result['weather_data']
    location = weather_result['location']
    coords = weather_result['coordinates']
    
    print("\n" + "="*70)
    print(f"🌤️  WEATHER FOR: {location}")
    
    # Place details
    if 'types' in place_info:
        place_types = ', '.join([t.replace('_', ' ').title() for t in place_info['types'][:3]])
        print(f"📍 Type: {place_types}")
    
    print(f"🗺️  Coordinates: {coords['lat']:.4f}, {coords['lng']:.4f}")
    
    # Current weather
    if 'currentConditions' in weather_data:
        current = weather_data['currentConditions']
        print(f"\n🌡️  CURRENT CONDITIONS:")
        print(f"   Temperature: {current.get('temp', 'N/A')}°C (feels like {current.get('feelslike', 'N/A')}°C)")
        print(f"   Conditions: {current.get('conditions', 'N/A')}")
        print(f"   Humidity: {current.get('humidity', 'N/A')}%")
        print(f"   Wind Speed: {current.get('windspeed', 'N/A')} km/h")
        print(f"   Visibility: {current.get('visibility', 'N/A')} km")
        print(f"   Pressure: {current.get('pressure', 'N/A')} mb")
    
    # Forecast
    if show_forecast and 'days' in weather_data:
        forecast_days = weather_data['days'][:5]  # Show 5 days
        print(f"\n📅 5-DAY FORECAST:")
        
        for i, day in enumerate(forecast_days):
            day_name = "Today" if i == 0 else datetime.strptime(day['datetime'], '%Y-%m-%d').strftime('%A')
            print(f"\n   {day_name} ({day['datetime']}):")
            print(f"      🌡️  High: {day.get('tempmax', 'N/A')}°C | Low: {day.get('tempmin', 'N/A')}°C")
            print(f"      ☁️  Conditions: {day.get('conditions', 'N/A')}")
            print(f"      💧 Humidity: {day.get('humidity', 'N/A')}% | 🌧️  Rain: {day.get('precip', 'N/A')}mm")
            print(f"      💨 Wind: {day.get('windspeed', 'N/A')} km/h")
            print(f"      📝 {day.get('description', 'N/A')}")
    
    print("="*70)

def interactive_mode(service: GooglePlacesWeatherService):
    """Run the service in interactive mode"""
    print("🌦️  Welcome to the Interactive Weather Location Service!")
    print("Powered by Google Places API + Visual Crossing Weather API")
    print("="*60)
    
    while True:
        print("\n📋 Choose an option:")
        print("1. Search for a place and get weather")
        print("2. Get weather by coordinates")
        print("3. Search multiple places and compare weather")
        print("4. Exit")
        
        choice = input("\n🎯 Enter your choice (1-4): ").strip()
        
        if choice == '4':
            print("\n👋 Thanks for using the Weather Location Service! Goodbye!")
            break
        
        elif choice == '1':
            query = input("\n🔍 Enter place name or address: ").strip()
            if not query:
                print("❌ Please enter a valid location!")
                continue
            
            forecast_choice = input("📅 Include forecast? (y/n): ").strip().lower()
            forecast_days = 5 if forecast_choice in ['y', 'yes'] else 0
            
            print(f"\n⏳ Searching for '{query}' and fetching weather data...")
            results = service.search_and_get_weather(query, forecast_days)
            
            if results:
                print(f"\n✅ Found {len(results)} result(s):")
                for result in results:
                    print_place_weather(result, show_forecast=forecast_days > 0)
            else:
                print("❌ No places found!")
        
        elif choice == '2':
            try:
                lat = float(input("\n🌍 Enter latitude: ").strip())
                lng = float(input("🌍 Enter longitude: ").strip())
                
                forecast_choice = input("📅 Include forecast? (y/n): ").strip().lower()
                forecast_days = 5 if forecast_choice in ['y', 'yes'] else 0
                
                print(f"\n⏳ Fetching weather data for {lat:.4f}, {lng:.4f}...")
                result = service.get_weather_by_coordinates(lat, lng, forecast_days=forecast_days)
                
                # Create a dummy result structure for printing
                weather_result = {
                    'success': result['success'],
                    'location': result['location'],
                    'coordinates': result['coordinates'],
                    'weather_data': result.get('weather_data', {}),
                    'place_info': {},
                    'error': result.get('error')
                }
                
                print_place_weather(weather_result, show_forecast=forecast_days > 0)
                
            except ValueError:
                print("❌ Please enter valid numeric coordinates!")
        
        elif choice == '3':
            query = input("\n🔍 Enter search query to compare multiple places: ").strip()
            if not query:
                print("❌ Please enter a valid query!")
                continue
            
            print(f"\n⏳ Searching for places matching '{query}'...")
            results = service.search_and_get_weather(query, 0)  # Current weather only for comparison
            
            if results:
                print(f"\n✅ Weather comparison for {len(results)} places:")
                print("\n" + "="*80)
                print(f"{'Location':<35} {'Temperature':<12} {'Conditions':<20} {'Humidity':<10}")
                print("="*80)
                
                for result in results:
                    if result['success'] and 'currentConditions' in result['weather_data']:
                        current = result['weather_data']['currentConditions']
                        location = result['location'][:32] + "..." if len(result['location']) > 35 else result['location']
                        temp = f"{current.get('temp', 'N/A')}°C"
                        conditions = current.get('conditions', 'N/A')[:17] + "..." if len(current.get('conditions', '')) > 20 else current.get('conditions', 'N/A')
                        humidity = f"{current.get('humidity', 'N/A')}%"
                        
                        print(f"{location:<35} {temp:<12} {conditions:<20} {humidity:<10}")
                    else:
                        location = result['location'][:32] + "..." if len(result['location']) > 35 else result['location']
                        print(f"{location:<35} {'Error':<12} {'N/A':<20} {'N/A':<10}")
                
                print("="*80)
            else:
                print("❌ No places found!")
        
        else:
            print("❌ Invalid choice! Please enter 1, 2, 3, or 4.")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Weather Location Service')
    parser.add_argument('--weather-key', default='5T4RN5AM9XUUVEEHMCJNCDCC4',
                       help='Visual Crossing Weather API key')
    parser.add_argument('--google-key', default='AIzaSyAOVYRIgupAurZup5y1PRh8Ismb1A3lLao',
                       help='Google Maps API key')
    parser.add_argument('--search', '-s', help='Search for a place')
    parser.add_argument('--forecast', '-f', action='store_true', help='Include forecast')
    parser.add_argument('--lat', type=float, help='Latitude for coordinate-based search')
    parser.add_argument('--lng', type=float, help='Longitude for coordinate-based search')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize the service
    service = GooglePlacesWeatherService(args.weather_key, args.google_key)
    
    if args.interactive:
        interactive_mode(service)
    elif args.search:
        forecast_days = 5 if args.forecast else 0
        results = service.search_and_get_weather(args.search, forecast_days)
        
        for result in results:
            print_place_weather(result, show_forecast=args.forecast)
    
    elif args.lat is not None and args.lng is not None:
        forecast_days = 5 if args.forecast else 0
        result = service.get_weather_by_coordinates(args.lat, args.lng, forecast_days=forecast_days)
        
        weather_result = {
            'success': result['success'],
            'location': result['location'],
            'coordinates': result['coordinates'],
            'weather_data': result.get('weather_data', {}),
            'place_info': {},
            'error': result.get('error')
        }
        
        print_place_weather(weather_result, show_forecast=args.forecast)
    
    else:
        # Default to interactive mode
        interactive_mode(service)

if __name__ == "__main__":
    main()