#!/usr/bin/env python3
"""
Quick Weather Checker
Usage: python quick_weather.py [location] [options]
Example: python quick_weather.py "London" --forecast 3
"""

import requests
import sys
import argparse

def get_weather_data(api_key, location, forecast_days=0):
    """Get weather data from Visual Crossing API"""
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    
    if forecast_days == 0:
        # Current weather only
        url = f"{base_url}/{location}/today"
        params = {
            'unitGroup': 'metric',
            'key': api_key,
            'include': 'current'
        }
    else:
        # Forecast
        url = f"{base_url}/{location}"
        params = {
            'unitGroup': 'metric',
            'key': api_key,
            'include': 'days',
            'elements': 'datetime,tempmax,tempmin,temp,humidity,precip,windspeed,conditions,description'
        }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching weather data: {e}")
        return None

def print_current_weather(data):
    """Print current weather information"""
    if 'currentConditions' not in data:
        print("❌ No current weather data available")
        return
    
    current = data['currentConditions']
    location = data.get('resolvedAddress', 'Unknown Location')
    
    print(f"\n🌤️  Current Weather for {location}")
    print("=" * 50)
    print(f"🌡️  Temperature: {current.get('temp', 'N/A')}°C (feels like {current.get('feelslike', 'N/A')}°C)")
    print(f"☁️  Conditions: {current.get('conditions', 'N/A')}")
    print(f"💧 Humidity: {current.get('humidity', 'N/A')}%")
    print(f"💨 Wind: {current.get('windspeed', 'N/A')} km/h")
    print(f"👁️  Visibility: {current.get('visibility', 'N/A')} km")
    print(f"📊 Pressure: {current.get('pressure', 'N/A')} mb")

def print_forecast(data, days):
    """Print weather forecast"""
    forecast_days = data.get('days', [])[:days]
    location = data.get('resolvedAddress', 'Unknown Location')
    
    print(f"\n📅 {days}-Day Forecast for {location}")
    print("=" * 60)
    
    for i, day in enumerate(forecast_days):
        date = day.get('datetime', 'Unknown')
        print(f"\n📆 {date}")
        print(f"  🌡️  High: {day.get('tempmax', 'N/A')}°C | Low: {day.get('tempmin', 'N/A')}°C")
        print(f"  ☁️  {day.get('conditions', 'N/A')}")
        print(f"  💧 Humidity: {day.get('humidity', 'N/A')}% | 💨 Wind: {day.get('windspeed', 'N/A')} km/h")
        print(f"  🌧️  Rain: {day.get('precip', 'N/A')} mm")
        
        if i < len(forecast_days) - 1:
            print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description='Quick Weather Checker using Visual Crossing API')
    parser.add_argument('location', help='Location to get weather for (e.g., "London", "New York")')
    parser.add_argument('--forecast', '-f', type=int, default=0, 
                       help='Number of forecast days (0 for current weather only)')
    parser.add_argument('--api-key', default='5T4RN5AM9XUUVEEHMCJNCDCC4',
                       help='Visual Crossing API key (default: provided key)')
    
    args = parser.parse_args()
    
    if not args.location:
        print("❌ Please provide a location!")
        sys.exit(1)
    
    print(f"⏳ Fetching weather data for {args.location}...")
    
    data = get_weather_data(args.api_key, args.location, args.forecast)
    
    if data is None:
        sys.exit(1)
    
    if args.forecast == 0:
        print_current_weather(data)
    else:
        print_forecast(data, args.forecast)

if __name__ == "__main__":
    main()