#!/usr/bin/env python3
"""
Enhanced Maharashtra AI Crop Forecasting System
Interactive Weather & Soil Analysis Dashboard with Plotly Charts
"""

from flask import Flask, render_template_string, jsonify, request
import json
import random
from datetime import datetime, timedelta
import math

app = Flask(__name__)

class EnhancedAgriculturalSystem:
    def __init__(self):
        """Initialize the enhanced agricultural system with interactive visualization"""
        
        # Maharashtra districts
        self.districts = [
            'Mumbai City', 'Mumbai Suburban', 'Pune', 'Nagpur', 'Thane',
            'Nashik', 'Chhatrapati Sambhajinagar', 'Solapur', 'Amravati',
            'Kolhapur', 'Sangli', 'Dhule', 'Jalgaon', 'Akola'
        ]
        
        # Crop types suitable for Maharashtra
        self.crop_types = [
            'Cotton', 'Rice', 'Wheat', 'Sugarcane', 'Soybean',
            'Tomato', 'Potato', 'Onion', 'Maize', 'Jowar'
        ]
    
    def get_weather_data(self, district):
        """Generate realistic weather data for Maharashtra districts"""
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)][::-1]
        
        # Maharashtra climate patterns
        base_temp = random.randint(25, 35)  # Celsius
        base_humidity = random.randint(60, 85)  # Percentage
        base_rainfall = random.randint(0, 25)  # mm
        base_wind = random.randint(5, 15)  # km/h
        
        weather_data = {
            'dates': dates,
            'temperature': [],
            'humidity': [],
            'rainfall': [],
            'wind_speed': [],
            'temperature_status': [],
            'humidity_status': [],
            'rainfall_status': []
        }
        
        for i in range(7):
            # Temperature with daily variation
            temp_variation = random.randint(-3, 3)
            temp = base_temp + temp_variation
            weather_data['temperature'].append(temp)
            
            # Temperature status categorization
            if temp > 32:
                weather_data['temperature_status'].append('Too Hot')
            elif temp < 22:
                weather_data['temperature_status'].append('Too Cold')
            else:
                weather_data['temperature_status'].append('Good')
            
            # Humidity with variation
            humidity_variation = random.randint(-10, 10)
            humidity = max(30, min(95, base_humidity + humidity_variation))
            weather_data['humidity'].append(humidity)
            
            # Humidity status
            if humidity > 80:
                weather_data['humidity_status'].append('High')
            elif humidity < 50:
                weather_data['humidity_status'].append('Low')
            else:
                weather_data['humidity_status'].append('Optimal')
            
            # Rainfall with variation
            rainfall_variation = random.randint(-5, 15)
            rainfall = max(0, base_rainfall + rainfall_variation)
            weather_data['rainfall'].append(rainfall)
            
            # Rainfall status
            if rainfall > 20:
                weather_data['rainfall_status'].append('Heavy')
            elif rainfall > 5:
                weather_data['rainfall_status'].append('Moderate')
            else:
                weather_data['rainfall_status'].append('Light')
            
            # Wind speed with variation
            wind_variation = random.randint(-3, 3)
            wind_speed = max(2, base_wind + wind_variation)
            weather_data['wind_speed'].append(wind_speed)
        
        return weather_data
    
    def create_interactive_weather_charts(self, weather_data):
        """Create interactive Plotly charts for weather data"""
        
        # Temperature Chart with Status Colors
        temp_colors = []
        for status in weather_data['temperature_status']:
            if status == 'Too Hot':
                temp_colors.append('#FF6B6B')
            elif status == 'Too Cold':
                temp_colors.append('#4ECDC4')
            else:
                temp_colors.append('#45B7D1')
        
        temperature_chart = {
            'data': [{
                'x': weather_data['dates'],
                'y': weather_data['temperature'],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Temperature (°C)',
                'line': {'color': '#FF6B6B', 'width': 3},
                'marker': {'color': temp_colors, 'size': 10, 'line': {'color': 'white', 'width': 2}},
                'text': [f"{temp}°C - {status}" for temp, status in zip(weather_data['temperature'], weather_data['temperature_status'])],
                'hovertemplate': '<b>%{text}</b><br>Date: %{x}<extra></extra>'
            }],
            'layout': {
                'title': {
                    'text': 'Temperature Trends - 7 Days',
                    'font': {'size': 18, 'color': '#2E86C1'},
                    'x': 0.5
                },
                'xaxis': {
                    'title': 'Date',
                    'gridcolor': '#E8F4F9',
                    'showgrid': True
                },
                'yaxis': {
                    'title': 'Temperature (°C)',
                    'gridcolor': '#E8F4F9',
                    'showgrid': True
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'font': {'color': '#2C3E50'},
                'height': 350,
                'showlegend': False,
                'margin': {'l': 60, 'r': 20, 't': 60, 'b': 50}
            }
        }
        
        # Humidity Chart with Status Colors
        humidity_colors = []
        for status in weather_data['humidity_status']:
            if status == 'High':
                humidity_colors.append('#E67E22')
            elif status == 'Low':
                humidity_colors.append('#F39C12')
            else:
                humidity_colors.append('#27AE60')
        
        humidity_chart = {
            'data': [{
                'x': weather_data['dates'],
                'y': weather_data['humidity'],
                'type': 'bar',
                'name': 'Humidity (%)',
                'marker': {
                    'color': humidity_colors,
                    'line': {'color': 'white', 'width': 1}
                },
                'text': [f"{hum}% - {status}" for hum, status in zip(weather_data['humidity'], weather_data['humidity_status'])],
                'hovertemplate': '<b>%{text}</b><br>Date: %{x}<extra></extra>'
            }],
            'layout': {
                'title': {
                    'text': 'Humidity Levels - 7 Days',
                    'font': {'size': 18, 'color': '#27AE60'},
                    'x': 0.5
                },
                'xaxis': {
                    'title': 'Date',
                    'gridcolor': '#E8F4F9',
                    'showgrid': True
                },
                'yaxis': {
                    'title': 'Humidity (%)',
                    'gridcolor': '#E8F4F9',
                    'showgrid': True,
                    'range': [0, 100]
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'font': {'color': '#2C3E50'},
                'height': 350,
                'showlegend': False,
                'margin': {'l': 60, 'r': 20, 't': 60, 'b': 50}
            }
        }
        
        # Rainfall Chart
        rainfall_colors = []
        for status in weather_data['rainfall_status']:
            if status == 'Heavy':
                rainfall_colors.append('#3498DB')
            elif status == 'Moderate':
                rainfall_colors.append('#5DADE2')
            else:
                rainfall_colors.append('#AED6F1')
        
        rainfall_chart = {
            'data': [{
                'x': weather_data['dates'],
                'y': weather_data['rainfall'],
                'type': 'bar',
                'name': 'Rainfall (mm)',
                'marker': {
                    'color': rainfall_colors,
                    'line': {'color': 'white', 'width': 1}
                },
                'text': [f"{rain}mm - {status}" for rain, status in zip(weather_data['rainfall'], weather_data['rainfall_status'])],
                'hovertemplate': '<b>%{text}</b><br>Date: %{x}<extra></extra>'
            }],
            'layout': {
                'title': {
                    'text': 'Rainfall Patterns - 7 Days',
                    'font': {'size': 18, 'color': '#3498DB'},
                    'x': 0.5
                },
                'xaxis': {
                    'title': 'Date',
                    'gridcolor': '#E8F4F9',
                    'showgrid': True
                },
                'yaxis': {
                    'title': 'Rainfall (mm)',
                    'gridcolor': '#E8F4F9',
                    'showgrid': True
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'font': {'color': '#2C3E50'},
                'height': 350,
                'showlegend': False,
                'margin': {'l': 60, 'r': 20, 't': 60, 'b': 50}
            }
        }
        
        # Wind Speed Chart
        wind_chart = {
            'data': [{
                'x': weather_data['dates'],
                'y': weather_data['wind_speed'],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Wind Speed (km/h)',
                'line': {'color': '#8E44AD', 'width': 3},
                'marker': {'color': '#9B59B6', 'size': 10, 'line': {'color': 'white', 'width': 2}},
                'fill': 'tozeroy',
                'fillcolor': 'rgba(142, 68, 173, 0.2)',
                'hovertemplate': '<b>%{y} km/h</b><br>Date: %{x}<extra></extra>'
            }],
            'layout': {
                'title': {
                    'text': 'Wind Speed Trends - 7 Days',
                    'font': {'size': 18, 'color': '#8E44AD'},
                    'x': 0.5
                },
                'xaxis': {
                    'title': 'Date',
                    'gridcolor': '#E8F4F9',
                    'showgrid': True
                },
                'yaxis': {
                    'title': 'Wind Speed (km/h)',
                    'gridcolor': '#E8F4F9',
                    'showgrid': True
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'font': {'color': '#2C3E50'},
                'height': 350,
                'showlegend': False,
                'margin': {'l': 60, 'r': 20, 't': 60, 'b': 50}
            }
        }
        
        return {
            'temperature': temperature_chart,
            'humidity': humidity_chart,
            'rainfall': rainfall_chart,
            'wind_speed': wind_chart
        }
    
    def get_soil_data(self, nitrogen=50, phosphorus=30, potassium=40, ph=6.5):
        """Generate soil analysis data"""
        
        # Calculate status based on optimal ranges
        n_status = 'Good' if 40 <= nitrogen <= 80 else ('Low' if nitrogen < 40 else 'High')
        p_status = 'Good' if 25 <= phosphorus <= 60 else ('Low' if phosphorus < 25 else 'High')
        k_status = 'Good' if 35 <= potassium <= 70 else ('Low' if potassium < 35 else 'High')
        ph_status = 'Good' if 6.0 <= ph <= 7.5 else ('Acidic' if ph < 6.0 else 'Alkaline')
        
        # Calculate overall soil health score
        n_score = min(100, max(0, (nitrogen / 80) * 100))
        p_score = min(100, max(0, (phosphorus / 60) * 100))
        k_score = min(100, max(0, (potassium / 70) * 100))
        ph_score = 100 - abs(6.8 - ph) * 15
        
        health_score = (n_score * 0.3 + p_score * 0.3 + k_score * 0.3 + ph_score * 0.1)
        
        if health_score >= 80:
            health_status = 'Excellent'
        elif health_score >= 70:
            health_status = 'Good'
        elif health_score >= 60:
            health_status = 'Fair'
        else:
            health_status = 'Poor'
        
        return {
            'nutrients': {
                'nitrogen': {'value': nitrogen, 'status': n_status, 'optimal_range': '40-80 ppm'},
                'phosphorus': {'value': phosphorus, 'status': p_status, 'optimal_range': '25-60 ppm'},
                'potassium': {'value': potassium, 'status': k_status, 'optimal_range': '35-70 ppm'},
                'ph': {'value': ph, 'status': ph_status, 'optimal_range': '6.0-7.5'}
            },
            'health_score': round(health_score, 1),
            'health_status': health_status,
            'scores': {
                'nitrogen_score': round(n_score, 1),
                'phosphorus_score': round(p_score, 1),
                'potassium_score': round(k_score, 1),
                'ph_score': round(ph_score, 1)
            }
        }
    
    def create_interactive_soil_charts(self, soil_data):
        """Create interactive Plotly charts for soil analysis"""
        
        nutrients = ['Nitrogen', 'Phosphorus', 'Potassium']
        values = [
            soil_data['nutrients']['nitrogen']['value'],
            soil_data['nutrients']['phosphorus']['value'],
            soil_data['nutrients']['potassium']['value']
        ]
        statuses = [
            soil_data['nutrients']['nitrogen']['status'],
            soil_data['nutrients']['phosphorus']['status'],
            soil_data['nutrients']['potassium']['status']
        ]
        
        # NPK Levels Chart
        npk_colors = []
        for status in statuses:
            if status == 'Good':
                npk_colors.append('#27AE60')
            elif status == 'Low':
                npk_colors.append('#E74C3C')
            else:
                npk_colors.append('#F39C12')
        
        npk_chart = {
            'data': [{
                'x': nutrients,
                'y': values,
                'type': 'bar',
                'name': 'NPK Levels (ppm)',
                'marker': {
                    'color': npk_colors,
                    'line': {'color': 'white', 'width': 2}
                },
                'text': [f"{val} ppm<br>{status}" for val, status in zip(values, statuses)],
                'hovertemplate': '<b>%{x}</b><br>%{text}<extra></extra>'
            }],
            'layout': {
                'title': {
                    'text': 'NPK Nutrient Levels Analysis',
                    'font': {'size': 18, 'color': '#27AE60'},
                    'x': 0.5
                },
                'xaxis': {
                    'title': 'Nutrients',
                    'gridcolor': '#E8F4F9'
                },
                'yaxis': {
                    'title': 'Concentration (ppm)',
                    'gridcolor': '#E8F4F9',
                    'showgrid': True
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'font': {'color': '#2C3E50'},
                'height': 350,
                'showlegend': False,
                'margin': {'l': 60, 'r': 20, 't': 60, 'b': 50}
            }
        }
        
        # Soil Health Score Gauge
        health_score = soil_data['health_score']
        
        gauge_chart = {
            'data': [{
                'type': 'indicator',
                'mode': 'gauge+number+delta',
                'value': health_score,
                'domain': {'x': [0, 1], 'y': [0, 1]},
                'title': {'text': f"Overall Soil Health<br><span style='font-size:0.8em;color:gray'>{soil_data['health_status']}</span>"},
                'gauge': {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': '#2E86C1'},
                    'steps': [
                        {'range': [0, 50], 'color': '#FADBD8'},
                        {'range': [50, 70], 'color': '#F9E79F'},
                        {'range': [70, 85], 'color': '#D5F4E6'},
                        {'range': [85, 100], 'color': '#A9DFBF'}
                    ],
                    'threshold': {
                        'line': {'color': '#E74C3C', 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            }],
            'layout': {
                'height': 350,
                'font': {'size': 16, 'color': '#2C3E50'},
                'paper_bgcolor': 'white',
                'margin': {'l': 20, 'r': 20, 't': 60, 'b': 20}
            }
        }
        
        # pH Analysis Chart
        ph_value = soil_data['nutrients']['ph']['value']
        ph_status = soil_data['nutrients']['ph']['status']
        
        ph_color = '#27AE60' if ph_status == 'Good' else ('#E74C3C' if ph_status == 'Acidic' else '#F39C12')
        
        ph_chart = {
            'data': [{
                'x': ['pH Level'],
                'y': [ph_value],
                'type': 'bar',
                'name': 'pH Level',
                'marker': {
                    'color': ph_color,
                    'line': {'color': 'white', 'width': 2}
                },
                'text': [f"pH {ph_value}<br>{ph_status}"],
                'hovertemplate': '<b>%{text}</b><br>Optimal: 6.0-7.5<extra></extra>'
            },
            {
                'x': ['pH Level'],
                'y': [6.0],
                'type': 'scatter',
                'mode': 'markers',
                'name': 'Optimal Min',
                'marker': {'color': '#D5DBDB', 'size': 12, 'symbol': 'line-ew'},
                'hovertemplate': 'Optimal Minimum: 6.0<extra></extra>'
            },
            {
                'x': ['pH Level'],
                'y': [7.5],
                'type': 'scatter',
                'mode': 'markers',
                'name': 'Optimal Max',
                'marker': {'color': '#D5DBDB', 'size': 12, 'symbol': 'line-ew'},
                'hovertemplate': 'Optimal Maximum: 7.5<extra></extra>'
            }],
            'layout': {
                'title': {
                    'text': 'Soil pH Analysis',
                    'font': {'size': 18, 'color': '#8E44AD'},
                    'x': 0.5
                },
                'xaxis': {
                    'title': 'Measurement',
                    'gridcolor': '#E8F4F9'
                },
                'yaxis': {
                    'title': 'pH Value',
                    'gridcolor': '#E8F4F9',
                    'showgrid': True,
                    'range': [4, 9]
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'font': {'color': '#2C3E50'},
                'height': 350,
                'showlegend': True,
                'legend': {'x': 1, 'y': 1},
                'margin': {'l': 60, 'r': 60, 't': 60, 'b': 50}
            }
        }
        
        # Nutrient Score Comparison
        score_chart = {
            'data': [{
                'labels': ['Nitrogen', 'Phosphorus', 'Potassium', 'pH'],
                'values': [
                    soil_data['scores']['nitrogen_score'],
                    soil_data['scores']['phosphorus_score'],
                    soil_data['scores']['potassium_score'],
                    soil_data['scores']['ph_score']
                ],
                'type': 'pie',
                'hole': 0.4,
                'marker': {
                    'colors': ['#3498DB', '#E67E22', '#9B59B6', '#F1C40F']
                },
                'textinfo': 'label+percent',
                'hovertemplate': '<b>%{label}</b><br>Score: %{value:.1f}<extra></extra>'
            }],
            'layout': {
                'title': {
                    'text': 'Nutrient Score Distribution',
                    'font': {'size': 18, 'color': '#34495E'},
                    'x': 0.5
                },
                'height': 350,
                'font': {'color': '#2C3E50'},
                'paper_bgcolor': 'white',
                'margin': {'l': 20, 'r': 20, 't': 60, 'b': 20}
            }
        }
        
        return {
            'npk_levels': npk_chart,
            'health_gauge': gauge_chart,
            'ph_analysis': ph_chart,
            'nutrient_distribution': score_chart
        }

# Initialize the system
agri_system = EnhancedAgriculturalSystem()

# Dashboard HTML Template
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maharashtra AI Crop Forecasting - Weather & Soil Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #2C3E50;
        }
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 20px 0;
            text-align: center;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2E86C1;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            color: #5D6D7E;
            font-size: 1.2em;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .tabs {
            display: flex;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px 15px 0 0;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .tab {
            flex: 1;
            padding: 20px;
            background: rgba(255, 255, 255, 0.8);
            border: none;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: 600;
            color: #5D6D7E;
            transition: all 0.3s ease;
        }
        .tab.active {
            background: #2E86C1;
            color: white;
        }
        .tab:hover:not(.active) {
            background: rgba(46, 134, 193, 0.1);
        }
        .tab-content {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 0 0 15px 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .controls {
            background: rgba(52, 152, 219, 0.1);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .control-group label {
            font-weight: 600;
            color: #34495E;
        }
        .control-group select, .control-group input {
            padding: 10px 15px;
            border: 2px solid #BDC3C7;
            border-radius: 8px;
            font-size: 16px;
            min-width: 150px;
            transition: border-color 0.3s ease;
        }
        .control-group select:focus, .control-group input:focus {
            outline: none;
            border-color: #2E86C1;
        }
        .btn-primary {
            background: #2E86C1;
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            align-self: end;
        }
        .btn-primary:hover {
            background: #21618C;
            transform: translateY(-2px);
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(650px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .chart-container:hover {
            transform: translateY(-5px);
        }
        .loading {
            text-align: center;
            padding: 50px;
            color: #5D6D7E;
            font-size: 1.2em;
        }
        .error {
            background: #FFE5E5;
            color: #C0392B;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #F1948A;
            margin: 20px 0;
        }
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .summary-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 3px 15px rgba(0,0,0,0.1);
            text-align: center;
        }
        .summary-card h3 {
            color: #2E86C1;
            margin-bottom: 10px;
        }
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #27AE60;
        }
        .summary-card .unit {
            color: #7F8C8D;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌾 Maharashtra AI Crop Forecasting System</h1>
        <p>Advanced Weather & Soil Analysis Dashboard</p>
    </div>
    
    <div class="container">
        <div class="tabs">
            <button class="tab active" onclick="showTab('weather')">Weather Analysis</button>
            <button class="tab" onclick="showTab('soil')">Soil Analysis</button>
        </div>
        
        <!-- Weather Analysis Tab -->
        <div id="weather" class="tab-content active">
            <div class="controls">
                <div class="control-group">
                    <label for="district-select">Select District:</label>
                    <select id="district-select">
                        <option value="Pune">Pune</option>
                        <option value="Mumbai City">Mumbai City</option>
                        <option value="Nagpur">Nagpur</option>
                        <option value="Nashik">Nashik</option>
                        <option value="Thane">Thane</option>
                        <option value="Chhatrapati Sambhajinagar">Chhatrapati Sambhajinagar</option>
                        <option value="Solapur">Solapur</option>
                        <option value="Amravati">Amravati</option>
                        <option value="Kolhapur">Kolhapur</option>
                        <option value="Sangli">Sangli</option>
                        <option value="Dhule">Dhule</option>
                        <option value="Jalgaon">Jalgaon</option>
                        <option value="Akola">Akola</option>
                    </select>
                </div>
                <button class="btn-primary" onclick="loadWeatherAnalysis()">Analyze Weather</button>
            </div>
            
            <div id="weather-summary" class="summary-cards" style="display: none;">
                <div class="summary-card">
                    <h3>Average Temperature</h3>
                    <div class="value" id="avg-temp">--</div>
                    <div class="unit">°C</div>
                </div>
                <div class="summary-card">
                    <h3>Average Humidity</h3>
                    <div class="value" id="avg-humidity">--</div>
                    <div class="unit">%</div>
                </div>
                <div class="summary-card">
                    <h3>Total Rainfall</h3>
                    <div class="value" id="total-rainfall">--</div>
                    <div class="unit">mm</div>
                </div>
                <div class="summary-card">
                    <h3>Average Wind Speed</h3>
                    <div class="value" id="avg-wind">--</div>
                    <div class="unit">km/h</div>
                </div>
            </div>
            
            <div id="weather-loading" class="loading" style="display: none;">
                Loading weather analysis...
            </div>
            
            <div id="weather-error" class="error" style="display: none;"></div>
            
            <div id="weather-charts" class="charts-grid">
                <div class="chart-container">
                    <div id="temperature-chart"></div>
                </div>
                <div class="chart-container">
                    <div id="humidity-chart"></div>
                </div>
                <div class="chart-container">
                    <div id="rainfall-chart"></div>
                </div>
                <div class="chart-container">
                    <div id="windspeed-chart"></div>
                </div>
            </div>
        </div>
        
        <!-- Soil Analysis Tab -->
        <div id="soil" class="tab-content">
            <div class="controls">
                <div class="control-group">
                    <label for="nitrogen-input">Nitrogen (ppm):</label>
                    <input type="number" id="nitrogen-input" value="50" min="0" max="150">
                </div>
                <div class="control-group">
                    <label for="phosphorus-input">Phosphorus (ppm):</label>
                    <input type="number" id="phosphorus-input" value="30" min="0" max="100">
                </div>
                <div class="control-group">
                    <label for="potassium-input">Potassium (ppm):</label>
                    <input type="number" id="potassium-input" value="40" min="0" max="120">
                </div>
                <div class="control-group">
                    <label for="ph-input">pH Level:</label>
                    <input type="number" id="ph-input" value="6.5" min="4.0" max="9.0" step="0.1">
                </div>
                <button class="btn-primary" onclick="loadSoilAnalysis()">Analyze Soil</button>
            </div>
            
            <div id="soil-summary" class="summary-cards" style="display: none;">
                <div class="summary-card">
                    <h3>Soil Health Score</h3>
                    <div class="value" id="soil-health-score">--</div>
                    <div class="unit" id="soil-health-status">--</div>
                </div>
                <div class="summary-card">
                    <h3>Nitrogen Status</h3>
                    <div class="value" id="nitrogen-status">--</div>
                    <div class="unit" id="nitrogen-value">-- ppm</div>
                </div>
                <div class="summary-card">
                    <h3>Phosphorus Status</h3>
                    <div class="value" id="phosphorus-status">--</div>
                    <div class="unit" id="phosphorus-value">-- ppm</div>
                </div>
                <div class="summary-card">
                    <h3>pH Level</h3>
                    <div class="value" id="ph-status">--</div>
                    <div class="unit" id="ph-value">--</div>
                </div>
            </div>
            
            <div id="soil-loading" class="loading" style="display: none;">
                Loading soil analysis...
            </div>
            
            <div id="soil-error" class="error" style="display: none;"></div>
            
            <div id="soil-charts" class="charts-grid">
                <div class="chart-container">
                    <div id="npk-chart"></div>
                </div>
                <div class="chart-container">
                    <div id="health-gauge-chart"></div>
                </div>
                <div class="chart-container">
                    <div id="ph-chart"></div>
                </div>
                <div class="chart-container">
                    <div id="nutrient-distribution-chart"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Tab functionality
        function showTab(tabName) {
            const tabs = document.querySelectorAll('.tab');
            const tabContents = document.querySelectorAll('.tab-content');
            
            tabs.forEach(tab => tab.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        
        // Weather Analysis Functions
        async function loadWeatherAnalysis() {
            const district = document.getElementById('district-select').value;
            const loadingEl = document.getElementById('weather-loading');
            const errorEl = document.getElementById('weather-error');
            const summaryEl = document.getElementById('weather-summary');
            
            loadingEl.style.display = 'block';
            errorEl.style.display = 'none';
            summaryEl.style.display = 'none';
            
            try {
                const response = await fetch(`/api/weather-analysis/${district}`);
                const data = await response.json();
                
                if (data.status === 'success') {
                    displayWeatherCharts(data.charts);
                    displayWeatherSummary(data.weather_data);
                    summaryEl.style.display = 'grid';
                } else {
                    throw new Error(data.message || 'Failed to load weather data');
                }
            } catch (error) {
                errorEl.textContent = 'Error loading weather analysis: ' + error.message;
                errorEl.style.display = 'block';
            } finally {
                loadingEl.style.display = 'none';
            }
        }
        
        function displayWeatherCharts(charts) {
            Plotly.newPlot('temperature-chart', charts.temperature.data, charts.temperature.layout, {responsive: true});
            Plotly.newPlot('humidity-chart', charts.humidity.data, charts.humidity.layout, {responsive: true});
            Plotly.newPlot('rainfall-chart', charts.rainfall.data, charts.rainfall.layout, {responsive: true});
            Plotly.newPlot('windspeed-chart', charts.wind_speed.data, charts.wind_speed.layout, {responsive: true});
        }
        
        function displayWeatherSummary(weatherData) {
            const avgTemp = weatherData.temperature.reduce((a, b) => a + b, 0) / weatherData.temperature.length;
            const avgHumidity = weatherData.humidity.reduce((a, b) => a + b, 0) / weatherData.humidity.length;
            const totalRainfall = weatherData.rainfall.reduce((a, b) => a + b, 0);
            const avgWind = weatherData.wind_speed.reduce((a, b) => a + b, 0) / weatherData.wind_speed.length;
            
            document.getElementById('avg-temp').textContent = avgTemp.toFixed(1);
            document.getElementById('avg-humidity').textContent = avgHumidity.toFixed(1);
            document.getElementById('total-rainfall').textContent = totalRainfall.toFixed(1);
            document.getElementById('avg-wind').textContent = avgWind.toFixed(1);
        }
        
        // Soil Analysis Functions
        async function loadSoilAnalysis() {
            const nitrogen = parseFloat(document.getElementById('nitrogen-input').value);
            const phosphorus = parseFloat(document.getElementById('phosphorus-input').value);
            const potassium = parseFloat(document.getElementById('potassium-input').value);
            const ph = parseFloat(document.getElementById('ph-input').value);
            
            const loadingEl = document.getElementById('soil-loading');
            const errorEl = document.getElementById('soil-error');
            const summaryEl = document.getElementById('soil-summary');
            
            loadingEl.style.display = 'block';
            errorEl.style.display = 'none';
            summaryEl.style.display = 'none';
            
            try {
                const response = await fetch('/api/soil-analysis', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ nitrogen, phosphorus, potassium, ph })
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    displaySoilCharts(data.charts);
                    displaySoilSummary(data.soil_data);
                    summaryEl.style.display = 'grid';
                } else {
                    throw new Error(data.message || 'Failed to load soil data');
                }
            } catch (error) {
                errorEl.textContent = 'Error loading soil analysis: ' + error.message;
                errorEl.style.display = 'block';
            } finally {
                loadingEl.style.display = 'none';
            }
        }
        
        function displaySoilCharts(charts) {
            Plotly.newPlot('npk-chart', charts.npk_levels.data, charts.npk_levels.layout, {responsive: true});
            Plotly.newPlot('health-gauge-chart', charts.health_gauge.data, charts.health_gauge.layout, {responsive: true});
            Plotly.newPlot('ph-chart', charts.ph_analysis.data, charts.ph_analysis.layout, {responsive: true});
            Plotly.newPlot('nutrient-distribution-chart', charts.nutrient_distribution.data, charts.nutrient_distribution.layout, {responsive: true});
        }
        
        function displaySoilSummary(soilData) {
            document.getElementById('soil-health-score').textContent = soilData.health_score;
            document.getElementById('soil-health-status').textContent = soilData.health_status;
            document.getElementById('nitrogen-status').textContent = soilData.nutrients.nitrogen.status;
            document.getElementById('nitrogen-value').textContent = soilData.nutrients.nitrogen.value + ' ppm';
            document.getElementById('phosphorus-status').textContent = soilData.nutrients.phosphorus.status;
            document.getElementById('phosphorus-value').textContent = soilData.nutrients.phosphorus.value + ' ppm';
            document.getElementById('ph-status').textContent = soilData.nutrients.ph.status;
            document.getElementById('ph-value').textContent = soilData.nutrients.ph.value;
        }
        
        // Load default weather analysis on page load
        document.addEventListener('DOMContentLoaded', function() {
            loadWeatherAnalysis();
        });
    </script>
</body>
</html>
'''

@app.route('/')
def dashboard():
    """Main dashboard route"""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/weather-analysis/<district>')
def weather_analysis(district):
    """Get weather analysis with interactive charts"""
    try:
        weather_data = agri_system.get_weather_data(district)
        charts = agri_system.create_interactive_weather_charts(weather_data)
        
        return jsonify({
            'status': 'success',
            'district': district,
            'weather_data': weather_data,
            'charts': charts
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/soil-analysis', methods=['POST'])
def soil_analysis():
    """Get soil analysis with interactive charts"""
    try:
        data = request.get_json()
        nitrogen = data.get('nitrogen', 50)
        phosphorus = data.get('phosphorus', 30)
        potassium = data.get('potassium', 40)
        ph = data.get('ph', 6.5)
        
        soil_data = agri_system.get_soil_data(nitrogen, phosphorus, potassium, ph)
        charts = agri_system.create_interactive_soil_charts(soil_data)
        
        return jsonify({
            'status': 'success',
            'soil_data': soil_data,
            'charts': charts
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("🌾 Enhanced Maharashtra Agricultural System Starting...")
    print("📊 Interactive Weather & Soil Analysis Dashboard")
    print("🔗 Dashboard: http://localhost:5004")
    print("✨ Features: Interactive Plotly Charts | No Hindi Words | Clean English UI")
    print("📈 Weather: Temperature, Humidity, Rainfall, Wind Speed Analysis")
    print("🌱 Soil: NPK Levels, pH Analysis, Health Scoring with Visual Indicators")
    print("-" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5004)