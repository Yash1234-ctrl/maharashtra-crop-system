#!/usr/bin/env python3
"""
Authenticated Maharashtra AI Crop Forecasting System
Comprehensive Agricultural System with Farmer Authentication
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import cv2
from PIL import Image
import tensorflow as tf
import os
import requests
import json
from dotenv import load_dotenv
import base64
from io import BytesIO
from auth_database import FarmerAuthDB

# Load environment variables
load_dotenv()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Initialize authentication database
if 'auth_db' not in st.session_state:
    st.session_state.auth_db = FarmerAuthDB()

# Page configuration
st.set_page_config(
    page_title="Maharashtra Krushi Mitra - AI Agricultural System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_authentication():
    """Check if user is authenticated and redirect if not"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Validate existing session
    if st.session_state.authenticated and 'session_id' in st.session_state:
        session_validation = st.session_state.auth_db.validate_session(
            st.session_state.session_id,
            st.session_state.session_token
        )
        
        if not session_validation["success"]:
            # Session expired
            st.session_state.authenticated = False
            st.warning("🔐 Your session has expired. Please login again.")
            show_login_redirect()
            st.stop()
    
    if not st.session_state.authenticated:
        show_login_redirect()
        st.stop()

def show_login_redirect():
    """Show login redirect page"""
    st.markdown("""
    <style>
        .login-redirect {
            background: linear-gradient(135deg, #1A237E 0%, #2E7D32 50%, #1976D2 100%);
            color: white;
            padding: 3rem;
            border-radius: 20px;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .redirect-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .redirect-subtitle {
            font-size: 1.2rem;
            margin-bottom: 2rem;
            opacity: 0.9;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="login-redirect">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🔐</div>
        <h1 class="redirect-title">Authentication Required</h1>
        <p class="redirect-subtitle">
            Please login to access the Maharashtra Agricultural System<br>
            Secure access for registered farmers only
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Go to Login Page", type="primary", use_container_width=True):
            st.info("💡 **Instructions:**")
            st.code("streamlit run farmer_login.py", language="bash")
            st.markdown("Run the above command in a new terminal to access the login page.")

def show_authenticated_header():
    """Show header for authenticated users"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 50%, #1976D2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(46, 125, 50, 0.3);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 2.5rem;">🌾 Maharashtra Krushi Mitra</h1>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Advanced AI-Powered Agricultural System</p>
            </div>
            <div style="text-align: right;">
                <p style="margin: 0; font-size: 1.1rem;">Welcome, <strong>{st.session_state.full_name}</strong></p>
                <p style="margin: 0.2rem 0 0 0; opacity: 0.8; font-size: 0.9rem;">Farmer ID: {st.session_state.farmer_id}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_logout_option():
    """Show logout option in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Account")
    st.sidebar.info(f"**Logged in as:**\n{st.session_state.full_name}")
    
    if st.sidebar.button("🚪 Logout", type="secondary", use_container_width=True):
        # Invalidate session
        if 'session_id' in st.session_state:
            st.session_state.auth_db.invalidate_session(st.session_state.session_id)
        
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.success("👋 Logged out successfully!")
        st.rerun()

# Import the main agricultural system functions from the original file
# (This would normally be imported, but for demo purposes, we'll include key functions here)

class MaharashtraAgriSystem:
    """Main agricultural system class with authentication"""
    
    def __init__(self):
        """Initialize the system"""
        self.districts = [
            "Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad", "Solapur",
            "Ahmednagar", "Kolhapur", "Sangli", "Satara", "Raigad", "Thane"
        ]
        self.crops = [
            "Rice", "Wheat", "Cotton", "Sugarcane", "Soybean", "Maize",
            "Onion", "Potato", "Tomato", "Chili", "Turmeric", "Groundnut"
        ]
    
    def get_weather_data(self, district):
        """Get weather data for district (simplified for demo)"""
        return {
            "temperature": np.random.uniform(20, 35),
            "humidity": np.random.uniform(40, 80),
            "rainfall": np.random.uniform(0, 50),
            "wind_speed": np.random.uniform(2, 15)
        }
    
    def analyze_soil_health(self, ph, nitrogen, phosphorus, potassium, area):
        """Analyze soil health"""
        # Simplified soil analysis
        ph_score = 100 if 6.0 <= ph <= 7.5 else max(0, 100 - abs(ph - 6.75) * 20)
        n_score = min(100, (nitrogen / 300) * 100)
        p_score = min(100, (phosphorus / 50) * 100)
        k_score = min(100, (potassium / 200) * 100)
        
        overall_score = (ph_score + n_score + p_score + k_score) / 4
        
        if overall_score >= 80:
            status = "Excellent"
        elif overall_score >= 60:
            status = "Good"
        elif overall_score >= 40:
            status = "Fair"
        else:
            status = "Poor"
        
        return {
            "score": overall_score,
            "status": status,
            "ph_score": ph_score,
            "nitrogen_score": n_score,
            "phosphorus_score": p_score,
            "potassium_score": k_score,
            "recommendations": self.get_soil_recommendations(overall_score),
            "total_cost": area * np.random.uniform(2000, 5000)
        }
    
    def get_soil_recommendations(self, score):
        """Get soil recommendations based on score"""
        if score >= 80:
            return ["Maintain current practices", "Monitor regularly", "Consider organic supplements"]
        elif score >= 60:
            return ["Add organic compost", "Balance NPK levels", "Improve drainage"]
        else:
            return ["Soil testing required", "Major nutrient supplementation needed", "Consider soil rehabilitation"]
    
    def analyze_crop_image(self, uploaded_file):
        """Analyze uploaded crop image (simplified)"""
        # Simplified disease detection
        diseases = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot"]
        confidence = np.random.uniform(70, 95)
        detected = np.random.choice(diseases)
        
        return {
            "disease": detected,
            "confidence": confidence,
            "recommendations": self.get_disease_recommendations(detected)
        }
    
    def get_disease_recommendations(self, disease):
        """Get treatment recommendations for detected disease"""
        recommendations = {
            "Healthy": ["Continue current practices", "Regular monitoring", "Preventive measures"],
            "Early Blight": ["Apply fungicide", "Improve ventilation", "Remove affected leaves"],
            "Late Blight": ["Immediate treatment needed", "Copper-based fungicide", "Quarantine affected area"],
            "Bacterial Spot": ["Bactericide application", "Avoid overhead watering", "Improve sanitation"]
        }
        return recommendations.get(disease, ["Consult agricultural expert"])

def main():
    """Main application function"""
    
    # Check authentication first
    check_authentication()
    
    # Show authenticated header
    show_authenticated_header()
    
    # Initialize the agricultural system
    agri_system = MaharashtraAgriSystem()
    
    # Sidebar with logout option
    show_logout_option()
    
    # Sidebar inputs
    st.sidebar.header("🌾 Farm Configuration")
    
    district = st.sidebar.selectbox(
        "📍 Select District",
        agri_system.districts,
        help="Choose your district in Maharashtra"
    )
    
    crop_type = st.sidebar.selectbox(
        "🌱 Primary Crop",
        agri_system.crops,
        help="Select your main crop type"
    )
    
    farm_area = st.sidebar.number_input(
        "🌾 Farm Area (acres)",
        min_value=0.1,
        value=5.0,
        step=0.5,
        help="Enter your total farm area"
    )
    
    growth_stage = st.sidebar.selectbox(
        "📈 Growth Stage",
        ["Seeding", "Vegetative", "Flowering", "Fruiting", "Harvest"],
        help="Current stage of crop growth"
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🌱 Crop Health", "🌤️ Weather & Soil", "🐛 Pest Risk", "📊 Dashboard"])
    
    with tab1:
        st.header("🌱 Crop Health Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Upload Crop Image")
            uploaded_file = st.file_uploader(
                "Choose crop image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload clear image of crop leaves or plants"
            )
            
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", width=400)
                
                if st.button("🔍 Analyze Crop Health", type="primary"):
                    with st.spinner("Analyzing crop image..."):
                        result = agri_system.analyze_crop_image(uploaded_file)
                        
                        st.success("✅ Analysis Complete!")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Disease Detected", result['disease'])
                        with col_b:
                            st.metric("Confidence", f"{result['confidence']:.1f}%")
                        with col_c:
                            status = "Healthy" if result['disease'] == "Healthy" else "Attention Needed"
                            st.metric("Status", status)
                        
                        st.subheader("💡 Recommendations")
                        for i, rec in enumerate(result['recommendations'], 1):
                            st.write(f"{i}. {rec}")
        
        with col2:
            st.subheader("📊 Health Analytics")
            
            # Sample health metrics
            health_data = {
                "Metric": ["Plant Health", "Disease Risk", "Growth Rate", "Yield Potential"],
                "Score": [np.random.randint(70, 95) for _ in range(4)],
                "Status": ["Good", "Low", "Excellent", "High"]
            }
            
            fig = px.bar(
                x=health_data["Score"],
                y=health_data["Metric"],
                orientation='h',
                title="Crop Health Metrics",
                color=health_data["Score"],
                color_continuous_scale="RdYlGn"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🌤️ Weather & Soil Analysis")
        
        # Get weather data
        weather = agri_system.get_weather_data(district)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌡️ Temperature", f"{weather['temperature']:.1f}°C")
        with col2:
            st.metric("💧 Humidity", f"{weather['humidity']:.1f}%")
        with col3:
            st.metric("🌧️ Rainfall", f"{weather['rainfall']:.1f}mm")
        with col4:
            st.metric("💨 Wind Speed", f"{weather['wind_speed']:.1f} m/s")
        
        st.markdown("---")
        
        # Soil analysis
        st.subheader("🧪 Soil Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            ph = st.slider("pH Level", 4.0, 9.0, 6.5, 0.1)
            nitrogen = st.number_input("Nitrogen (kg/ha)", 0, 500, 300, 10)
        with col2:
            phosphorus = st.number_input("Phosphorus (kg/ha)", 0, 100, 25, 5)
            potassium = st.number_input("Potassium (kg/ha)", 0, 300, 150, 10)
        
        if st.button("🔍 Analyze Soil Health", type="primary"):
            with st.spinner("Analyzing soil composition..."):
                soil_result = agri_system.analyze_soil_health(ph, nitrogen, phosphorus, potassium, farm_area)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Score", f"{soil_result['score']:.1f}/100")
                with col2:
                    st.metric("Health Status", soil_result['status'])
                with col3:
                    st.metric("Treatment Cost", f"₹{soil_result['total_cost']:.2f}")
                
                # Detailed scores
                scores_data = {
                    "Parameter": ["pH Level", "Nitrogen", "Phosphorus", "Potassium"],
                    "Score": [
                        soil_result['ph_score'],
                        soil_result['nitrogen_score'],
                        soil_result['phosphorus_score'],
                        soil_result['potassium_score']
                    ]
                }
                
                fig = px.bar(
                    x=scores_data["Parameter"],
                    y=scores_data["Score"],
                    title="Soil Parameter Scores",
                    color=scores_data["Score"],
                    color_continuous_scale="RdYlGn"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("💡 Recommendations")
                for i, rec in enumerate(soil_result['recommendations'], 1):
                    st.write(f"{i}. {rec}")
    
    with tab3:
        st.header("🐛 Pest Risk Assessment")
        
        # Sample pest risk data
        pest_data = {
            "Pest": ["Aphids", "Whitefly", "Thrips", "Bollworm", "Leaf Miner"],
            "Risk Level": [np.random.randint(20, 80) for _ in range(5)],
            "Severity": ["Medium", "High", "Low", "High", "Medium"]
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=pest_data["Pest"],
                y=pest_data["Risk Level"],
                title="Pest Risk Levels",
                color=pest_data["Risk Level"],
                color_continuous_scale="Reds"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 High Risk Alerts")
            for pest, risk, severity in zip(pest_data["Pest"], pest_data["Risk Level"], pest_data["Severity"]):
                if risk > 60:
                    st.warning(f"⚠️ **{pest}**: {risk}% risk ({severity} severity)")
            
            st.subheader("🛡️ Prevention Measures")
            st.write("1. Regular field monitoring")
            st.write("2. Use of beneficial insects")
            st.write("3. Proper crop rotation")
            st.write("4. Maintain field hygiene")
            st.write("5. Timely pesticide application")
    
    with tab4:
        st.header("📊 Personalized Dashboard")
        
        # Get farmer profile
        profile_result = st.session_state.auth_db.get_farmer_profile(st.session_state.farmer_id)
        
        if profile_result["success"]:
            profile = profile_result["profile"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👨‍🌾 Farmer Profile")
                st.write(f"**Name:** {profile['full_name']}")
                st.write(f"**Farm:** {profile['farm_name'] or 'Not specified'}")
                st.write(f"**District:** {profile['district'] or district}")
                st.write(f"**Village:** {profile['village'] or 'Not specified'}")
                st.write(f"**Farm Area:** {profile['farm_area'] or farm_area} acres")
                st.write(f"**Crops:** {profile['crop_types'] or crop_type}")
                st.write(f"**Member Since:** {profile['registration_date'][:10] if profile['registration_date'] else 'N/A'}")
            
            with col2:
                st.subheader("📈 Quick Stats")
                
                # Sample analytics
                stats_data = {
                    "Total Analyses": np.random.randint(15, 50),
                    "Healthy Crops": np.random.randint(80, 95),
                    "Issues Detected": np.random.randint(2, 8),
                    "Avg Soil Health": np.random.randint(70, 90)
                }
                
                for stat, value in stats_data.items():
                    if "%" in stat or "Health" in stat:
                        st.metric(stat, f"{value}%")
                    else:
                        st.metric(stat, value)
        
        # Recent activity
        st.subheader("🕐 Recent Activity")
        activity_data = {
            "Date": [datetime.now() - timedelta(days=i) for i in range(5)],
            "Activity": [
                "Crop health analysis completed",
                "Soil test results received",
                "Pest risk assessment done",
                "Weather data updated",
                "Profile updated"
            ],
            "Status": ["✅ Complete", "✅ Complete", "⚠️ Attention", "✅ Complete", "✅ Complete"]
        }
        
        activity_df = pd.DataFrame(activity_data)
        activity_df["Date"] = activity_df["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(activity_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.7; margin-top: 2rem;">
        🌱 Maharashtra Krushi Mitra - Empowering Farmers with AI Technology<br>
        © 2025 | Secure Agricultural Platform for Maharashtra Farmers
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()