#!/usr/bin/env python3
"""
Farmer Login Page
Beautiful agricultural-themed authentication system for Maharashtra Agricultural System
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import base64
from io import BytesIO
import hashlib
from auth_database import FarmerAuthDB
import os
from streamlit_option_menu import option_menu

# Configure the login page
st.set_page_config(
    page_title="Maharashtra Krushi Mitra - Farmer Login",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize the authentication database
if 'auth_db' not in st.session_state:
    st.session_state.auth_db = FarmerAuthDB()

# Enhanced Agricultural Theme with Modern UI
st.markdown("""
<style>
    /* === ROOT VARIABLES === */
:root {
    --card-bg: #0f172a;
    --card-surface: #1e293b;
    --accent: #22c55e;
    --accent-dark: #15803d;
    --text: #e6eef0;
    --text-secondary: #94a3b8;
    --card-shadow: 0 10px 30px rgba(2,6,23,0.7);
    --input-bg: rgba(15, 23, 42, 0.8);
    --success: #15803d;
    --success-gradient: linear-gradient(135deg, #22c55e 0%, #15803d 100%);
    --warning: #854d0e;
    --danger: #991b1b;
    --border-radius: 1rem;
}

/* === PAGE LAYOUT === */
.stApp {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    min-height: 100vh;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 800px;
}

/* === HIDE STREAMLIT ELEMENTS === */
#MainMenu, footer, header, [data-testid="stToolbar"] {
    display: none !important;
}

.stApp > div:first-child {
    margin-top: -80px;
}
    
    /* === CORE STYLES === */
    .main {
        background: linear-gradient(180deg,#081018 0%, #0e1117 100%);
        color: var(--text);
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* === LOGIN CARD === */
    .login-card {
        background-color: var(--card-surface);
        padding: 2.5rem;
        border-radius: 1.5rem;
        box-shadow: var(--card-shadow);
        max-width: 480px;
        margin: 1rem auto;
        position: relative;
    }

    /* === TOP BANNER === */
    .top-banner {
        text-align: center;
        background-color: var(--accent);
        padding: 1rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        max-width: 1100px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* === FORM ELEMENTS === */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: var(--input-bg) !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: var(--border-radius) !important;
    color: var(--text) !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.2s ease !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(34,197,94,0.2) !important;
}

/* Form Labels */
.stTextInput > label,
.stTextArea > label {
    color: var(--text-secondary) !important;
    font-size: 0.95rem !important;
}

/* Buttons */
button[kind="primary"] {
    background: var(--success-gradient) !important;
    border: none !important;
    border-radius: var(--border-radius) !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s ease !important;
}

button[kind="primary"]:hover {
    box-shadow: 0 4px 15px rgba(34,197,94,0.4) !important;
    transform: translateY(-2px) !important;
}

button[kind="primary"]:active {
    transform: translateY(0) !important;
}

/* Secondary Buttons */
button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    border-radius: var(--border-radius) !important;
    color: var(--accent) !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
}

button[kind="secondary"]:hover {
    background: rgba(34,197,94,0.1) !important;
}

/* Checkboxes */
.stCheckbox > label > div[role="checkbox"] {
    border-color: var(--accent) !important;
    border-radius: 4px !important;
}

.stCheckbox > label > div[data-checked="true"] {
    background-color: var(--accent) !important;
}

    /* === OPTION MENU === */
    .option-menu {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }

    .nav-link {
        color: var(--text) !important;
        background: var(--card-bg) !important;
        border-radius: 8px !important;
        margin: 0 0.5rem !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
    }

    .nav-link.active {
        background: var(--success-gradient) !important;
        color: white !important;
        transform: translateY(-2px) !important;
    }
    
    /* === SUCCESS/ERROR MESSAGES === */
    .stSuccess {
        background: linear-gradient(135deg, var(--secondary-green), var(--accent-green));
        color: white;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .stError {
        background: linear-gradient(135deg, var(--danger-red), #FF5252);
        color: white;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
    }
    
    .stWarning {
        background: linear-gradient(135deg, var(--warning-amber), var(--sunshine-yellow));
        color: white;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
    }
    
    /* === SELECT BOX STYLING === */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(76, 175, 80, 0.3);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--secondary-green);
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
    }
    
    /* === FOOTER STYLES === */
    .login-footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(76, 175, 80, 0.2);
    }
    
    .footer-text {
        color: var(--earth-brown);
        font-size: 0.9rem;
        opacity: 0.7;
    }
    
    /* === SUCCESS/ERROR === */
    .stSuccess {
        background: var(--success-gradient) !important;
        color: white !important;
        border: none !important;
        padding: 1rem !important;
    }

    .stError {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%) !important;
        color: white !important;
        border: none !important;
    }

    /* === RESPONSIVE === */
    @media (max-width: 600px) {
        .login-card { 
            padding: 1.5rem;
            max-width: calc(100% - 2rem);
        }
        .top-banner {
            margin-bottom: 1rem;
            padding: 0.6rem;
        }
        h1 { font-size: 1.4rem; }
    }

    @media (max-width: 420px) {
        .stApp .option-menu > div > div {
            flex-direction: column !important;
        }
    }

    @media (max-width: 400px) {
        .top-banner { display: none; }
    }

    /* === HIDE STREAMLIT === */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .stApp > header { display: none !important; }
</style>
""", unsafe_allow_html=True)

def show_login_page():
    """Display the enhanced login/registration interface"""
    
    # Top banner (hides on small screens via CSS)
    st.markdown(
        """
        <div class="top-banner">
            <h1 style="color:white; margin:0;">🌾 MahaAgroAI</h1>
            <div style="color:white; opacity:0.95; font-size:0.95rem">Secure Farmer Access Portal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs using option_menu
    selected = option_menu(
        menu_title=None,
        options=["Login", "Register"],
        icons=["lock-fill", "person-plus-fill"],
        orientation="horizontal",
        styles={
            "container": {"justify-content": "center"},
            "icon": {"color": "#22c55e", "font-size": "18px"},
            "nav-link-selected": {"background-color": "#22c55e", "color": "white"}
        }
    )

    if selected == "Login":
        show_login_form()
    else:
        show_registration_form()

def show_login_form():
    """Display the enhanced login form with modern UI"""
    
    st.markdown(
        """
        <div class="form-header">
            <h2>Welcome Back, Kisan! 🌾</h2>
            <p>Access your smart farming tools</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("login_form", clear_on_submit=False):
        # Username field with icon
        username = st.text_input(
            "Username or Email",
            placeholder="Enter your username or email",
            help="Use the email or username you registered with"
        )
        
        # Password with visibility toggle
        password_col, toggle_col = st.columns([4,1])
        with password_col:
            password = st.text_input(
                "Password",
                type="password" if not st.session_state.get('show_password', False) else "text",
                placeholder="Enter your password",
                help="Enter your password"
            )
        with toggle_col:
            st.checkbox(
                "👁️",
                key="show_password",
                help="Show/hide password"
            )
        
        # Remember me and submit button
        remember = st.checkbox(
            "Keep me signed in",
            help="Stay logged in for 7 days"
        )
        
        submit_button = st.form_submit_button(
            "Sign In",
            use_container_width=True,
            type="primary"
        )
        
        # Current season indicator with modern styling
        current_month = datetime.now().month
        if 6 <= current_month <= 9:
            season = "Kharif Season 🌧️"
            season_desc = "Monitor monsoon crops"
        elif 10 <= current_month <= 2:
            season = "Rabi Season ❄️"
            season_desc = "Track winter crops"
        else:
            season = "Zaid Season ☀️"
            season_desc = "Manage summer crops"
            
        st.markdown(
            f"""
            <div style="padding:1rem; background:var(--card-bg); border-radius:0.8rem; margin-top:1rem;">
                <div style="font-size:0.9rem; opacity:0.8;">Current Growing Period</div>
                <div style="font-size:1.2rem; font-weight:600; margin-top:0.3rem;">{season}</div>
                <div style="font-size:0.85rem; opacity:0.7; margin-top:0.2rem;">{season_desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Process login
    if submit_button:
        if not username or not password:
            st.error("Please enter both username/email and password")
            return
            
        with st.spinner("Authenticating..."):
            auth_result = st.session_state.auth_db.authenticate_farmer(
                username, password, ip_address="127.0.0.1"
            )
            
            if auth_result["success"]:
                session_result = st.session_state.auth_db.create_session(
                    auth_result["farmer_id"],
                    ip_address="127.0.0.1",
                    user_agent="Streamlit Browser"
                )
                
                if session_result["success"]:
                    st.session_state.authenticated = True
                    st.session_state.farmer_id = auth_result["farmer_id"]
                    st.session_state.username = auth_result["username"]
                    st.session_state.full_name = auth_result["full_name"]
                    st.session_state.session_id = session_result["session_id"]
                    st.session_state.session_token = session_result["session_token"]
                    
                    st.success("✅ Login successful!")
                    st.info("🌾 Loading your agricultural dashboard...")
                    st.experimental_rerun()
                else:
                    st.error("Session creation failed. Please try again.")
            else:
                st.error(auth_result["message"])

    # Quick access buttons with modern styling
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔑 Reset Password", use_container_width=True):
            st.info("Password reset coming soon!")
    with col2:
        if st.button("❓ Need Help?", use_container_width=True):
            st.info("Contact: support@maharashtra-krushi.gov.in")
            
    # Feature highlights
    st.markdown(
        """
        <div style="margin-top:2rem;">
            <div style="font-size:0.9rem; opacity:0.8; margin-bottom:1rem;">Available Tools:</div>
            <div class="feature-grid" style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;">
                <div style="background:var(--card-bg); padding:1rem; border-radius:0.8rem;">
                    <div style="font-size:1.2rem;">🌱 Disease Detection</div>
                    <div style="font-size:0.85rem; opacity:0.7; margin-top:0.3rem;">AI-powered crop health analysis</div>
                </div>
                <div style="background:var(--card-bg); padding:1rem; border-radius:0.8rem;">
                    <div style="font-size:1.2rem;">🌦️ Weather Monitor</div>
                    <div style="font-size:0.85rem; opacity:0.7; margin-top:0.3rem;">Real-time weather insights</div>
                </div>
                <div style="background:var(--card-bg); padding:1rem; border-radius:0.8rem;">
                    <div style="font-size:1.2rem;">🚜 Resource Planner</div>
                    <div style="font-size:0.85rem; opacity:0.7; margin-top:0.3rem;">Optimize farm operations</div>
                </div>
                <div style="background:var(--card-bg); padding:1rem; border-radius:0.8rem;">
                    <div style="font-size:1.2rem;">📊 Farm Analytics</div>
                    <div style="font-size:0.85rem; opacity:0.7; margin-top:0.3rem;">Data-driven insights</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def show_registration_form():
    """Registration form for new farmers"""
    st.markdown("### 📝 Join Our Farming Community!")
    st.markdown("Create your account to access advanced agricultural tools")
    
    with st.form("registration_form"):
        # Basic Information
        st.markdown("#### 👤 Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input(
                "👨‍🌾 Full Name *",
                placeholder="Enter your full name",
                help="Your complete name as per government records"
            )
            
            username = st.text_input(
                "👤 Username *",
                placeholder="Choose a unique username",
                help="This will be your login identifier"
            )
            
            phone = st.text_input(
                "📱 Phone Number",
                placeholder="Enter your mobile number",
                help="For SMS alerts and support"
            )
        
        with col2:
            email = st.text_input(
                "📧 Email Address *",
                placeholder="Enter your email",
                help="For important notifications and password recovery"
            )
            
            password = st.text_input(
                "🔒 Password *",
                type="password",
                placeholder="Create a strong password",
                help="Minimum 6 characters recommended"
            )
            
            confirm_password = st.text_input(
                "🔒 Confirm Password *",
                type="password",
                placeholder="Re-enter your password",
                help="Must match the password above"
            )
        
        # Farm Information
        st.markdown("#### 🌾 Farm Information")
        col1, col2 = st.columns(2)
        
        with col1:
            farm_name = st.text_input(
                "🏡 Farm Name",
                placeholder="Enter your farm name",
                help="Optional: Name of your farm or land"
            )
            
            district = st.selectbox(
                "📍 District *",
                ["Select District", "Pune", "Mumbai", "Nagpur", "Nashik", "Aurangabad", 
                 "Solapur", "Ahmednagar", "Kolhapur", "Sangli", "Satara", "Raigad", 
                 "Thane", "Nandurbar", "Dhule", "Jalgaon", "Buldhana", "Akola", 
                 "Washim", "Amravati", "Wardha", "Yavatmal", "Gadchiroli", "Chandrapur", 
                 "Gondia", "Bhandara", "Nagpur", "Latur", "Osmanabad", "Beed", 
                 "Parbhani", "Hingoli", "Nanded", "Jalna", "Ratnagiri", "Sindhudurg"],
                help="Select your district in Maharashtra"
            )
            
            farm_area = st.number_input(
                "🌾 Farm Area (acres)",
                min_value=0.0,
                value=0.0,
                step=0.5,
                help="Total cultivated area in acres"
            )
        
        with col2:
            village = st.text_input(
                "🏘️ Village/City",
                placeholder="Enter your village or city",
                help="Your village or city name"
            )
            
            # Enhanced crop selection with categorization
            crop_categories = {
                "Kharif Crops 🌧️": ["Rice", "Cotton", "Soybean", "Maize", "Jowar", "Bajra", "Tur/Arhar", "Moong", "Urad"],
                "Rabi Crops ❄️": ["Wheat", "Chana", "Mustard", "Peas", "Potato"],
                "Cash Crops 💰": ["Sugarcane", "Cotton", "Sunflower", "Turmeric"],
                "Vegetables 🥬": ["Onion", "Potato", "Tomato", "Chili", "Brinjal", "Cauliflower"],
                "Fruits �果": ["Mango", "Banana", "Grapes", "Pomegranate", "Orange"]
            }
            
            selected_category = st.selectbox(
                "🌱 Crop Category",
                options=list(crop_categories.keys()),
                help="Choose your primary crop category"
            )
            
            crop_types = st.multiselect(
                "🌾 Select Crops",
                options=crop_categories[selected_category],
                help="Select the specific crops you grow in this category"
            )
        
        # Terms and Conditions
        st.markdown("#### 📋 Terms & Agreement")
        terms_accepted = st.checkbox(
            "✅ I accept the Terms of Service and Privacy Policy",
            help="You must accept to create an account"
        )
        
        newsletter = st.checkbox(
            "📧 Subscribe to agricultural updates and tips",
            value=True,
            help="Get latest farming techniques and weather alerts"
        )
        
        # Register button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            register_button = st.form_submit_button(
                "🚀 CREATE MY ACCOUNT",
                use_container_width=True
            )
    
    # Process registration
    if register_button:
        # Validation
        errors = []
        
        if not full_name or len(full_name) < 2:
            errors.append("Full name is required (minimum 2 characters)")
        
        if not username or len(username) < 3:
            errors.append("Username is required (minimum 3 characters)")
        
        if not email or "@" not in email:
            errors.append("Valid email address is required")
        
        if not password or len(password) < 6:
            errors.append("Password must be at least 6 characters")
        
        if password != confirm_password:
            errors.append("Passwords do not match")
        
        if district == "Select District":
            errors.append("Please select your district")
        
        if not terms_accepted:
            errors.append("You must accept the terms and conditions")
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        else:
            with st.spinner("📝 Creating your farmer account..."):
                # Register farmer
                registration_result = st.session_state.auth_db.register_farmer(
                    username=username,
                    email=email,
                    password=password,
                    full_name=full_name,
                    phone=phone,
                    farm_name=farm_name,
                    district=district,
                    village=village,
                    farm_area=farm_area,
                    crop_types=", ".join(crop_types) if crop_types else ""
                )
                
                if registration_result["success"]:
                    st.success("✅ Account created successfully!")
                    st.info("🚪 You can now login with your credentials in the Login tab.")
                    
                    # Show welcome message
                    st.markdown(f"""
                    <div class="welcome-card">
                        <h2 class="welcome-title">Welcome to Maharashtra Krushi Mitra!</h2>
                        <p class="welcome-subtitle">
                            Hello <strong>{full_name}</strong>!<br>
                            Your account has been successfully created.<br>
                            Farmer ID: <strong>{registration_result['farmer_id']}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.error(f"❌ Registration failed: {registration_result['message']}")

def show_dashboard():
    """Show the main agricultural dashboard for authenticated farmers"""
    # Get current time for greeting
    current_hour = datetime.now().hour
    if current_hour < 12:
        greeting = "Good Morning"
    elif current_hour < 16:
        greeting = "Good Afternoon"
    else:
        greeting = "Good Evening"

    st.markdown(f"""
    <div class="welcome-card">
        <h1 class="welcome-title">� {greeting}, {st.session_state.full_name}!</h1>
        <p class="welcome-subtitle">
            Your AI-Powered Agricultural Assistant is Ready<br>
            <strong>किसान ID:</strong> {st.session_state.farmer_id} | 
            <strong>उपयोगकर्ता:</strong> {st.session_state.username}<br>
            <small style='opacity: 0.8;'>Last login: {datetime.now().strftime('%d %B %Y, %I:%M %p')}</small>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Logout button
    if st.button("🚪 Logout", type="secondary"):
        # Invalidate session
        if 'session_id' in st.session_state:
            st.session_state.auth_db.invalidate_session(st.session_state.session_id)
        
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.success("👋 Logged out successfully!")
        st.rerun()
    
    st.markdown("---")
    st.info("🚀 **Ready to launch your Maharashtra Agricultural System!**")
    
    # Launch button to main system
    if st.button("🌾 LAUNCH AGRICULTURAL DASHBOARD", type="primary", use_container_width=True):
        st.success("🚀 Launching your personalized agricultural system...")
        st.info("💡 **Instructions:** Run your `maharashtra_crop_system.py` file to access the full system!")
        
        # Show command to run
        st.code("streamlit run maharashtra_crop_system.py", language="bash")

# Main application logic
def main():
    """Main application entry point"""
    
    # Check if user is authenticated
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Validate existing session
    if st.session_state.authenticated and 'session_id' in st.session_state:
        session_validation = st.session_state.auth_db.validate_session(
            st.session_state.session_id,
            st.session_state.session_token
        )
        
        if not session_validation["success"]:
            # Session expired or invalid
            st.session_state.authenticated = False
            for key in ['farmer_id', 'username', 'full_name', 'session_id', 'session_token']:
                if key in st.session_state:
                    del st.session_state[key]
    
    # Show appropriate page
    if st.session_state.authenticated:
        show_dashboard()
    else:
        show_login_page()

if __name__ == "__main__":
    main()