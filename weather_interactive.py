import requests
import json
from datetime import datetime, timedelta

class VisualCrossingWeatherAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    
    def get_current_weather(self, location):
        """Get current weather for a location"""
        url = f"{self.base_url}/{location}/today"
        
        params = {
            'unitGroup': 'metric',
            'key': self.api_key,
            'include': 'current'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'currentConditions' in data:
                current = data['currentConditions']
                location = data.get('resolvedAddress', 'Unknown Location')
                
                return {
                    'success': True,
                    'location': location,
                    'datetime': current.get('datetime', 'Unknown'),
                    'temperature': current.get('temp', 'N/A'),
                    'feels_like': current.get('feelslike', 'N/A'),
                    'humidity': current.get('humidity', 'N/A'),
                    'conditions': current.get('conditions', 'N/A'),
                    'wind_speed': current.get('windspeed', 'N/A'),
                    'visibility': current.get('visibility', 'N/A'),
                    'pressure': current.get('pressure', 'N/A')
                }
            else:
                return {'success': False, 'error': 'No current weather data available'}
                
        except requests.exceptions.RequestException as e:
            return {'success': False, 'error': f"Error fetching weather data: {e}"}
    
    def get_weather_forecast(self, location, days=7):
        """Get weather forecast for specified days"""
        url = f"{self.base_url}/{location}"
        
        params = {
            'unitGroup': 'metric',
            'key': self.api_key,
            'include': 'days',
            'elements': 'datetime,tempmax,tempmin,temp,humidity,precip,windspeed,conditions,description'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            forecast_days = data.get('days', [])[:days]
            location = data.get('resolvedAddress', 'Unknown Location')
            
            forecast_info = {
                'success': True,
                'location': location,
                'forecast': []
            }
            
            for day in forecast_days:
                day_info = {
                    'date': day.get('datetime', 'Unknown'),
                    'max_temp': day.get('tempmax', 'N/A'),
                    'min_temp': day.get('tempmin', 'N/A'),
                    'humidity': day.get('humidity', 'N/A'),
                    'precipitation': day.get('precip', 'N/A'),
                    'wind_speed': day.get('windspeed', 'N/A'),
                    'conditions': day.get('conditions', 'N/A'),
                    'description': day.get('description', 'N/A')
                }
                forecast_info['forecast'].append(day_info)
            
            return forecast_info
            
        except requests.exceptions.RequestException as e:
            return {'success': False, 'error': f"Error fetching forecast data: {e}"}

def print_current_weather(weather_data):
    """Pretty print current weather data"""
    if not weather_data['success']:
        print(f"❌ {weather_data['error']}")
        return
    
    print("\n" + "🌤️ " + "="*48)
    print(f"CURRENT WEATHER FOR {weather_data['location'].upper()}")
    print("="*50)
    print(f"🕐 Time: {weather_data['datetime']}")
    print(f"🌡️  Temperature: {weather_data['temperature']}°C")
    print(f"🤔 Feels Like: {weather_data['feels_like']}°C")
    print(f"☁️  Conditions: {weather_data['conditions']}")
    print(f"💧 Humidity: {weather_data['humidity']}%")
    print(f"💨 Wind Speed: {weather_data['wind_speed']} km/h")
    print(f"👁️  Visibility: {weather_data['visibility']} km")
    print(f"📊 Pressure: {weather_data['pressure']} mb")
    print("="*50)

def print_forecast(forecast_data, days_to_show=None):
    """Pretty print forecast data"""
    if not forecast_data['success']:
        print(f"❌ {forecast_data['error']}")
        return
    
    forecast_list = forecast_data['forecast']
    if days_to_show:
        forecast_list = forecast_list[:days_to_show]
    
    print("\n" + "📅 " + "="*58)
    print(f"WEATHER FORECAST FOR {forecast_data['location'].upper()}")
    print("="*60)
    
    for i, day in enumerate(forecast_list):
        day_name = get_day_name(day['date'])
        print(f"\n📆 {day_name} ({day['date']})")
        print(f"  🌡️  {day['max_temp']}°C / {day['min_temp']}°C")
        print(f"  ☁️  {day['conditions']}")
        print(f"  💧 Humidity: {day['humidity']}% | 💨 Wind: {day['wind_speed']} km/h")
        print(f"  🌧️  Precipitation: {day['precipitation']} mm")
        print(f"  📝 {day['description']}")
        if i < len(forecast_list) - 1:
            print("-" * 40)

def get_day_name(date_str):
    """Convert date string to day name"""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        today = datetime.now().date()
        date_only = date_obj.date()
        
        if date_only == today:
            return "Today"
        elif date_only == today + timedelta(days=1):
            return "Tomorrow"
        else:
            return date_obj.strftime("%A")
    except:
        return "Unknown Day"

def main():
    # Initialize the weather API with your key
    API_KEY = "5T4RN5AM9XUUVEEHMCJNCDCC4"
    weather_api = VisualCrossingWeatherAPI(API_KEY)
    
    print("🌦️  Welcome to the Interactive Weather App!")
    print("Powered by Visual Crossing Weather API")
    print("="*50)
    
    while True:
        print("\n📋 Choose an option:")
        print("1. Current Weather")
        print("2. 3-Day Forecast")
        print("3. 7-Day Forecast")
        print("4. Exit")
        
        choice = input("\n🎯 Enter your choice (1-4): ").strip()
        
        if choice == '4':
            print("\n👋 Thanks for using the Weather App! Goodbye!")
            break
        
        if choice in ['1', '2', '3']:
            location = input("\n📍 Enter location (city, country): ").strip()
            
            if not location:
                print("❌ Please enter a valid location!")
                continue
            
            print(f"\n⏳ Fetching weather data for {location}...")
            
            if choice == '1':
                # Current weather
                weather_data = weather_api.get_current_weather(location)
                print_current_weather(weather_data)
                
            elif choice == '2':
                # 3-day forecast
                forecast_data = weather_api.get_weather_forecast(location, days=3)
                print_forecast(forecast_data)
                
            elif choice == '3':
                # 7-day forecast
                forecast_data = weather_api.get_weather_forecast(location, days=7)
                print_forecast(forecast_data)
        
        else:
            print("❌ Invalid choice! Please enter 1, 2, 3, or 4.")
        
        # Ask if user wants to continue
        continue_choice = input("\n❓ Would you like to check another location? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("\n👋 Thanks for using the Weather App! Goodbye!")
            break

if __name__ == "__main__":
    main()