#!/usr/bin/env python3
"""
Maharashtra AI Crop Forecasting System - Simplified Version
Main system file with temporary MongoDB replacement for testing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
from datetime import datetime, timedelta
import json
from io import BytesIO
import base64
import tensorflow as tf
import requests
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Import local modules - using simplified versions
from mongodb_config_simplified import MongoCropDB
from enhanced_pest_data import PEST_DATABASE, get_disease_severity

# Configure tensorflow to avoid memory issues
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Suppress plotly warnings
warnings.filterwarnings('ignore', message='.*keyword arguments.*')

# Page configuration
st.set_page_config(
    page_title="Maharashtra AI Crop Forecasting System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MaharashtraAgriculturalSystem:
    def __init__(self):
        """Initialize the system with simulated MongoDB"""
        self.setup_database()
        self.load_models()
        
        # API Keys
        self.openweather_api_key = os.getenv('OPENWEATHER_API_KEY', 'your_api_key_here')
        self.agromonitoring_api_key = os.getenv('AGROMONITORING_API_KEY', 'your_api_key_here')
        
        # Maharashtra districts and zones (all 36 districts)
        self.maharashtra_districts = {
            'Konkan (Coastal)': ['Mumbai City', 'Mumbai Suburban', 'Palghar', 'Thane', 'Raigad', 'Ratnagiri', 'Sindhudurg'],
            'Western Maharashtra': ['Pune', 'Satara', 'Sangli', 'Kolhapur', 'Solapur'],
            'North Maharashtra (Khandesh)': ['Nashik', 'Dhule', 'Nandurbar', 'Jalgaon', 'Ahmednagar'],
            'Marathwada': ['Chhatrapati Sambhajinagar', 'Jalna', 'Beed', 'Latur', 'Osmanabad', 'Nanded', 'Parbhani', 'Hingoli'],
            'Vidarbha': ['Nagpur', 'Wardha', 'Amravati', 'Akola', 'Washim', 'Buldhana', 'Yavatmal', 'Chandrapur', 'Gadchiroli', 'Bhandara', 'Gondia']
        }
        
        # District coordinates (for map visualization)
        self.district_coords = {
            'Mumbai City': (18.9388, 72.8354),
            'Mumbai Suburban': (19.1180, 72.9050),
            'Palghar': (19.6967, 72.7655),
            # ... [other coordinates remain the same]
        }
        
        # Crop types
        self.crop_types = [
            'Cotton', 'Rice', 'Wheat', 'Sugarcane', 'Soybean',
            'Tomato', 'Potato', 'Onion', 'Maize', 'Jowar'
        ]
        
        # Growth stages
        self.growth_stages = [
            'Sowing', 'Germination', 'Vegetative', 'Flowering',
            'Fruit Development', 'Maturity', 'Harvesting'
        ]
    
    def setup_database(self):
        """Setup simplified database connection"""
        self.mongo_db = MongoCropDB()
    
    def load_models(self):
        """Load AI models"""
        try:
            if os.path.exists('best_model.h5'):
                self.disease_model = tf.keras.models.load_model('best_model.h5')
                if os.path.exists('class_names.txt'):
                    with open('class_names.txt', 'r') as f:
                        self.class_names = [line.strip() for line in f.readlines()]
                else:
                    self.class_names = ['Healthy', 'Early_Blight', 'Late_Blight', 'Bacterial_Spot']
            else:
                self.disease_model = None
                self.class_names = ['Healthy', 'Early_Blight', 'Late_Blight', 'Bacterial_Spot']
        except Exception as e:
            self.disease_model = None
            self.class_names = ['Healthy', 'Early_Blight', 'Late_Blight', 'Bacterial_Spot']
            st.warning(f"Model loading error: {str(e)}")
    
    def run(self):
        """Main application logic"""
        st.title("🌾 Maharashtra AI Crop Forecasting System")
        st.subheader("Welcome to the Advanced Agricultural Assistant")
        
        # Main menu
        menu = st.sidebar.selectbox(
            "Choose System Feature",
            ["Home", "Crop Disease Analysis", "Weather Analysis", "Soil Health Assessment", 
             "Pest Risk Prediction", "Irrigation Management"]
        )
        
        if menu == "Home":
            self.show_home_page()
        elif menu == "Crop Disease Analysis":
            self.show_disease_analysis()
        elif menu == "Weather Analysis":
            self.show_weather_analysis()
        elif menu == "Soil Health Assessment":
            self.show_soil_analysis()
        elif menu == "Pest Risk Prediction":
            self.show_pest_prediction()
        elif menu == "Irrigation Management":
            self.show_irrigation_management()
    
    def show_home_page(self):
        """Display home page with system overview"""
        st.markdown("""
        ## Welcome to Maharashtra's Advanced Agricultural System
        
        This AI-powered platform helps farmers make data-driven decisions for better crop management.
        
        ### Key Features:
        - 🔬 **Crop Disease Detection** using AI image analysis
        - 🌦️ **Weather Monitoring** with precise forecasting
        - 🌱 **Soil Health Analysis** for optimal nutrition
        - 🐛 **Pest Risk Prediction** using advanced algorithms
        - 💧 **Smart Irrigation Management** for water efficiency
        
        ### Getting Started:
        1. Select a feature from the sidebar menu
        2. Follow the guided instructions
        3. Get AI-powered recommendations
        
        ### Current Weather Overview:
        """)
        
        # Show current weather for sample districts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            weather = self.get_current_weather("Mumbai City")
            st.metric("Mumbai", f"{weather['temperature']}°C", f"{weather['humidity']}% Humidity")
        
        with col2:
            weather = self.get_current_weather("Pune")
            st.metric("Pune", f"{weather['temperature']}°C", f"{weather['humidity']}% Humidity")
        
        with col3:
            weather = self.get_current_weather("Nagpur")
            st.metric("Nagpur", f"{weather['temperature']}°C", f"{weather['humidity']}% Humidity")
    
    def show_disease_analysis(self):
        """Display crop disease analysis interface"""
        st.markdown("""
        ## 🔬 Crop Disease Analysis
        
        Upload an image of your crop for AI-powered disease detection.
        """)
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    results = self.analyze_crop_image(uploaded_file)
                    
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        st.success(f"Analysis Complete: {results['disease']}")
                        st.info(f"Confidence: {results['confidence']:.1f}%")
                        
                        # Show all predictions
                        st.subheader("Detailed Analysis:")
                        for disease, prob in results['all_predictions']:
                            st.write(f"- {disease}: {prob:.1f}%")
                        
                        # Show recommendations
                        st.subheader("Recommendations:")
                        for rec in results['recommendations']:
                            st.markdown(rec)
    
    def show_weather_analysis(self):
        """Display weather analysis interface"""
        st.markdown("""
        ## 🌦️ Weather Analysis
        
        Get detailed weather information and forecasts for your district.
        """)
        
        # District selection
        district = st.selectbox("Select District", 
                              [d for districts in self.maharashtra_districts.values() for d in districts])
        
        if district:
            weather_data = self.get_weather_data(district)
            current = weather_data['current']
            forecast = weather_data['forecast']
            
            # Current weather
            st.subheader("Current Weather")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Temperature", f"{current['temperature']}°C")
            with col2:
                st.metric("Humidity", f"{current['humidity']}%")
            with col3:
                st.metric("Wind Speed", f"{current['wind_speed']} km/h")
            with col4:
                st.metric("Visibility", f"{current['visibility']} km")
            
            # Weather forecast
            st.subheader("5-Day Forecast")
            df_forecast = pd.DataFrame({
                'Date': forecast['dates'],
                'Temperature': forecast['temperature'],
                'Humidity': forecast['humidity'],
                'Rainfall': forecast['rainfall'],
                'Conditions': forecast['conditions']
            })
            
            st.dataframe(df_forecast)
            
            # Plot forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast['Temperature'],
                                   mode='lines+markers', name='Temperature'))
            fig.update_layout(title='Temperature Forecast', 
                            xaxis_title='Date',
                            yaxis_title='Temperature (°C)')
            st.plotly_chart(fig)
    
    def show_soil_analysis(self):
        """Display soil analysis interface"""
        st.markdown("""
        ## 🌱 Soil Health Assessment
        
        Analyze your soil conditions and get fertilizer recommendations.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            ph = st.slider("Soil pH", 0.0, 14.0, 7.0, 0.1)
            nitrogen = st.number_input("Nitrogen (kg/ha)", 0.0, 500.0, 200.0)
        
        with col2:
            phosphorus = st.number_input("Phosphorus (kg/ha)", 0.0, 300.0, 100.0)
            potassium = st.number_input("Potassium (kg/ha)", 0.0, 300.0, 150.0)
        
        if st.button("Analyze Soil Health"):
            results = self.analyze_soil_health(ph, nitrogen, phosphorus, potassium)
            
            # Display results
            st.subheader(f"Soil Health Score: {results['score']}/100")
            st.markdown(f"Status: **{results['status']}**")
            
            # Show recommendations
            st.subheader("Recommendations:")
            for rec in results['recommendations']:
                st.markdown(rec)
            
            # Show fertilizer recommendations
            st.subheader("Fertilizer Recommendations:")
            for fert in results['fertilizer_recommendations']:
                st.markdown(f"""
                * **{fert['type']}**
                  - Quantity: {fert['quantity']} kg/ha
                  - Cost: ₹{fert['cost']}
                  - Purpose: {fert['purpose']}
                  - Application: {fert['application_method']}
                """)
    
    def show_pest_prediction(self):
        """Display pest prediction interface"""
        st.markdown("""
        ## 🐛 Pest Risk Prediction
        
        Get advanced pest risk analysis based on weather conditions and crop type.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            district = st.selectbox("Select District",
                                  [d for districts in self.maharashtra_districts.values() for d in districts])
            crop_type = st.selectbox("Select Crop", self.crop_types)
        
        with col2:
            growth_stage = st.selectbox("Growth Stage", self.growth_stages)
        
        if st.button("Analyze Pest Risk"):
            weather_data = self.get_weather_data(district)
            results = self.analyze_pest_risk(weather_data, crop_type, growth_stage)
            
            # Display risk score
            st.subheader(f"Overall Risk Score: {results['overall_risk']}/100")
            st.markdown(f"Risk Level: **{results['risk_level']['level']}**")
            
            # Show individual risk factors
            st.subheader("Risk Factors:")
            for factor, data in results['risk_factors'].items():
                st.markdown(f"- **{factor.title()}**: {data['status']} ({data['value']}%)")
            
            # Show pest predictions
            st.subheader("Predicted Pests:")
            for pest in results['pest_predictions']:
                st.markdown(f"""
                * **{pest['pest']}**
                  - Probability: {pest['probability']}%
                  - Severity: {pest['severity']}
                """)
            
            # Show recommendations
            st.subheader("Management Recommendations:")
            for rec in results['recommendations']:
                st.markdown(rec)
    
    def show_irrigation_management(self):
        """Display irrigation management interface"""
        st.markdown("""
        ## 💧 Smart Irrigation Management
        
        Get personalized irrigation recommendations based on crop, weather, and soil conditions.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            district = st.selectbox("Select District",
                                  [d for districts in self.maharashtra_districts.values() for d in districts])
            crop_type = st.selectbox("Select Crop", self.crop_types)
            growth_stage = st.selectbox("Growth Stage", self.growth_stages)
        
        with col2:
            soil_ph = st.slider("Soil pH", 0.0, 14.0, 7.0, 0.1)
            farm_area = st.number_input("Farm Area (Hectares)", 0.1, 100.0, 1.0)
        
        if st.button("Get Irrigation Recommendations"):
            weather = self.get_current_weather(district)
            recommendations = self.get_irrigation_recommendations(
                crop_type, district, growth_stage, soil_ph, farm_area, weather
            )
            
            # Display recommendations
            st.subheader("Irrigation Requirements:")
            st.markdown(f"""
            - Daily Water Requirement: **{recommendations['daily_water_requirement']} mm**
            - Water per Hectare: **{recommendations['water_per_hectare']} liters**
            - Recommended Frequency: **{recommendations['irrigation_frequency']}**
            - Suggested Method: **{recommendations['recommended_method']}**
            """)
            
            # Show critical stages
            st.subheader("Critical Growth Stages:")
            for stage in recommendations['critical_growth_stages']:
                st.markdown(f"- {stage}")

# Create and run the application
if __name__ == "__main__":
    app = MaharashtraAgriculturalSystem()
    app.run()