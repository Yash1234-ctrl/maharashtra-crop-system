import requests
import json
from datetime import datetime

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
            
            return self.format_current_weather(data)
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {e}"
    
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
            
            return self.format_forecast(data, days)
        except requests.exceptions.RequestException as e:
            return f"Error fetching forecast data: {e}"
    
    def get_historical_weather(self, location, start_date, end_date):
        """Get historical weather data"""
        url = f"{self.base_url}/{location}/{start_date}/{end_date}"
        
        params = {
            'unitGroup': 'metric',
            'key': self.api_key,
            'include': 'days',
            'elements': 'datetime,tempmax,tempmin,temp,humidity,precip,windspeed,conditions'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return self.format_historical(data)
        except requests.exceptions.RequestException as e:
            return f"Error fetching historical data: {e}"
    
    def format_current_weather(self, data):
        """Format current weather data for display"""
        if 'currentConditions' in data:
            current = data['currentConditions']
            location = data.get('resolvedAddress', 'Unknown Location')
            
            weather_info = {
                'location': location,
                'datetime': current.get('datetime', 'Unknown'),
                'temperature': f"{current.get('temp', 'N/A')}°C",
                'feels_like': f"{current.get('feelslike', 'N/A')}°C",
                'humidity': f"{current.get('humidity', 'N/A')}%",
                'conditions': current.get('conditions', 'N/A'),
                'wind_speed': f"{current.get('windspeed', 'N/A')} km/h",
                'visibility': f"{current.get('visibility', 'N/A')} km",
                'pressure': f"{current.get('pressure', 'N/A')} mb"
            }
            
            return weather_info
        else:
            return "No current weather data available"
    
    def format_forecast(self, data, days):
        """Format forecast data for display"""
        forecast_days = data.get('days', [])[:days]
        location = data.get('resolvedAddress', 'Unknown Location')
        
        forecast_info = {
            'location': location,
            'forecast': []
        }
        
        for day in forecast_days:
            day_info = {
                'date': day.get('datetime', 'Unknown'),
                'max_temp': f"{day.get('tempmax', 'N/A')}°C",
                'min_temp': f"{day.get('tempmin', 'N/A')}°C",
                'humidity': f"{day.get('humidity', 'N/A')}%",
                'precipitation': f"{day.get('precip', 'N/A')} mm",
                'wind_speed': f"{day.get('windspeed', 'N/A')} km/h",
                'conditions': day.get('conditions', 'N/A'),
                'description': day.get('description', 'N/A')
            }
            forecast_info['forecast'].append(day_info)
        
        return forecast_info
    
    def format_historical(self, data):
        """Format historical data for display"""
        historical_days = data.get('days', [])
        location = data.get('resolvedAddress', 'Unknown Location')
        
        historical_info = {
            'location': location,
            'historical_data': []
        }
        
        for day in historical_days:
            day_info = {
                'date': day.get('datetime', 'Unknown'),
                'max_temp': f"{day.get('tempmax', 'N/A')}°C",
                'min_temp': f"{day.get('tempmin', 'N/A')}°C",
                'avg_temp': f"{day.get('temp', 'N/A')}°C",
                'humidity': f"{day.get('humidity', 'N/A')}%",
                'precipitation': f"{day.get('precip', 'N/A')} mm",
                'wind_speed': f"{day.get('windspeed', 'N/A')} km/h",
                'conditions': day.get('conditions', 'N/A')
            }
            historical_info['historical_data'].append(day_info)
        
        return historical_info

def print_current_weather(weather_data):
    """Pretty print current weather data"""
    if isinstance(weather_data, str):
        print(weather_data)
        return
    
    print("\n" + "="*50)
    print(f"CURRENT WEATHER FOR {weather_data['location'].upper()}")
    print("="*50)
    print(f"Time: {weather_data['datetime']}")
    print(f"Temperature: {weather_data['temperature']}")
    print(f"Feels Like: {weather_data['feels_like']}")
    print(f"Conditions: {weather_data['conditions']}")
    print(f"Humidity: {weather_data['humidity']}")
    print(f"Wind Speed: {weather_data['wind_speed']}")
    print(f"Visibility: {weather_data['visibility']}")
    print(f"Pressure: {weather_data['pressure']}")
    print("="*50)

def print_forecast(forecast_data):
    """Pretty print forecast data"""
    if isinstance(forecast_data, str):
        print(forecast_data)
        return
    
    print("\n" + "="*60)
    print(f"WEATHER FORECAST FOR {forecast_data['location'].upper()}")
    print("="*60)
    
    for day in forecast_data['forecast']:
        print(f"\nDate: {day['date']}")
        print(f"  Max Temp: {day['max_temp']} | Min Temp: {day['min_temp']}")
        print(f"  Conditions: {day['conditions']}")
        print(f"  Humidity: {day['humidity']} | Wind: {day['wind_speed']}")
        print(f"  Precipitation: {day['precipitation']}")
        print(f"  Description: {day['description']}")
        print("-" * 40)

# Example usage
if __name__ == "__main__":
    # Initialize the weather API with your key
    API_KEY = "5T4RN5AM9XUUVEEHMCJNCDCC4"
    weather_api = VisualCrossingWeatherAPI(API_KEY)
    
    # Example 1: Get current weather
    print("Example 1: Current Weather")
    location = "New York, NY"  # You can change this to any location
    current_weather = weather_api.get_current_weather(location)
    print_current_weather(current_weather)
    
    # Example 2: Get 5-day forecast
    print("\n\nExample 2: 5-Day Forecast")
    forecast = weather_api.get_weather_forecast(location, days=5)
    print_forecast(forecast)
    
    # Example 3: Get historical weather (last week)
    print("\n\nExample 3: Historical Weather (sample)")
    historical = weather_api.get_historical_weather(location, "2024-01-01", "2024-01-07")
    if isinstance(historical, dict):
        print(f"Historical data for {historical['location']}:")
        for day in historical['historical_data'][:3]:  # Show first 3 days
            print(f"  {day['date']}: {day['avg_temp']} | {day['conditions']}")
    else:
        print(historical)