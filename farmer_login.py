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

# Beautiful Agricultural CSS with Enhanced Styling
st.markdown("""
<style>
    /* === GLOBAL VARIABLES === */
    :root {
        --primary-green: #1B5E20;
        --secondary-green: #388E3C;
        --accent-green: #43A047;
        --earth-brown: #3E2723;
        --soil-brown: #5D4037;
        --sky-blue: #0D47A1;
        --water-blue: #0277BD;
        --sunshine-yellow: #F57F17;
        --harvest-orange: #EF6C00;
        --danger-red: #C62828;
        --warning-amber: #FF8F00;
        --text-white: #FFFFFF;
        --text-dark: #1A1A1A;
        --bg-gradient: linear-gradient(135deg, #004D40 0%, #1B5E20 50%, #0D47A1 100%);
        --card-shadow: 0 20px 40px rgba(0,0,0,0.15);
        --border-radius: 20px;
        --success-gradient: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%);
        --warning-gradient: linear-gradient(135deg, #F57F17 0%, #FF8F00 100%);
        --danger-gradient: linear-gradient(135deg, #C62828 0%, #D32F2F 100%);
    }
    
    /* === HIDE STREAMLIT ELEMENTS === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp > div:first-child {margin-top: -80px;}
    
    /* === BODY & BACKGROUND === */
    .stApp {
        background: var(--bg-gradient);
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* === ANIMATED BACKGROUND === */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(76, 175, 80, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(25, 118, 210, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(255, 167, 38, 0.1) 0%, transparent 50%);
        animation: backgroundMove 20s ease-in-out infinite;
        z-index: -1;
    }
    
    @keyframes backgroundMove {
        0%, 100% { transform: translateX(0px) translateY(0px); }
        25% { transform: translateX(-20px) translateY(-10px); }
        50% { transform: translateX(20px) translateY(-20px); }
        75% { transform: translateX(-10px) translateY(10px); }
    }
    
    /* === MAIN CONTAINER === */
    .login-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: var(--border-radius);
        padding: 3rem;
        margin: 2rem auto;
        max-width: 480px;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .login-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-green), var(--secondary-green), var(--sky-blue));
    }
    
    /* === HEADER STYLES === */
    .login-header {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    .app-logo {
        font-size: 4rem;
        margin-bottom: 0.5rem;
        animation: bounce 2s ease-in-out infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-green);
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: var(--earth-brown);
        font-weight: 400;
        opacity: 0.8;
    }
    
    /* === FORM STYLES === */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(76, 175, 80, 0.3);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--secondary-green);
        box-shadow: 0 0 20px rgba(76, 175, 80, 0.2);
        transform: translateY(-2px);
    }
    
    /* === BUTTON STYLES === */
    .stButton > button {
        background: linear-gradient(135deg, var(--secondary-green) 0%, var(--primary-green) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(76, 175, 80, 0.4);
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--secondary-green) 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* === TABS STYLING === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(76, 175, 80, 0.1);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-radius: 8px;
        color: var(--primary-green);
        font-weight: 600;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--secondary-green);
        color: white;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
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
    
    /* === RESPONSIVE DESIGN === */
    @media (max-width: 768px) {
        .login-container {
            margin: 1rem;
            padding: 2rem;
            max-width: 100%;
        }
        
        .app-title {
            font-size: 1.6rem;
        }
        
        .app-subtitle {
            font-size: 1rem;
        }
    }
    
    /* === LOADING ANIMATION === */
    .stSpinner > div {
        border-color: var(--secondary-green) transparent var(--secondary-green) transparent;
    }
    
    /* === FORM VALIDATION === */
    .field-error {
        color: var(--danger-red);
        font-size: 0.9rem;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    
    /* === WELCOME MESSAGE === */
    .welcome-card {
        background: linear-gradient(135deg, var(--secondary-green), var(--accent-green));
        color: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        text-align: center;
        box-shadow: var(--card-shadow);
        margin: 2rem 0;
    }
    
    .welcome-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .welcome-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def show_login_page():
    """Display the main login/registration interface"""
    
    # Show weather alert if available
    current_month = datetime.now().month
    if 6 <= current_month <= 9:  # Monsoon season
        st.warning("""
            🌧️ **Monsoon Alert**
            - Keep track of rainfall patterns
            - Monitor crop disease risks
            - Ensure proper drainage
        """)
    elif 10 <= current_month <= 2:  # Winter season
        st.info("""
            ❄️ **Rabi Season Guidelines**
            - Watch for frost warnings
            - Maintain soil moisture
            - Monitor cold-weather crops
        """)
    else:  # Summer season
        st.warning("""
            ☀️ **Summer Advisory**
            - Practice water conservation
            - Watch for heat stress in crops
            - Consider crop insurance
        """)
    
    # Create main container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # Header section
    st.markdown("""
    <div class="login-header">
        <div class="app-logo">🌾</div>
        <h1 class="app-title">Maharashtra Krushi Mitra</h1>
        <p class="app-subtitle">Advanced AI-Powered Agricultural System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs for Login and Registration
    tab1, tab2 = st.tabs(["🚪 Login", "📝 Register"])
    
    with tab1:
        show_login_form()
    
    with tab2:
        show_registration_form()
    
    # Footer
    st.markdown("""
    <div class="login-footer">
        <p class="footer-text">
            🌱 Empowering Maharashtra Farmers with AI Technology<br>
            © 2025 Maharashtra Krushi Mitra | Secure & Reliable
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_login_form():
    """Login form for existing farmers"""
    st.markdown("### 🌾 Welcome Back, Kisan!")
    st.markdown("""
        Enter your credentials to access:
        - 🌱 AI-Powered Crop Disease Detection
        - 🌦️ Smart Weather Monitoring
        - 🚜 Agricultural Resource Planning
        - 📊 Farm Analytics Dashboard
    """)
    
    # Show current agricultural season
    current_month = datetime.now().month
    if 6 <= current_month <= 9:
        season = "Kharif Season 🌧️"
    elif 10 <= current_month <= 2:
        season = "Rabi Season ❄️"
    else:
        season = "Zaid Season ☀️"
    
    st.info(f"Current Growing Period: **{season}**")
    
    with st.form("login_form"):
        # Username/Email field
        username = st.text_input(
            "👤 Username or Email",
            placeholder="Enter your username or email",
            help="Use the username or email you registered with"
        )
        
        # Password field
        password = st.text_input(
            "🔒 Password",
            type="password",
            placeholder="Enter your password",
            help="Password is case-sensitive"
        )
        
        # Remember me checkbox
        remember_me = st.checkbox("🔄 Keep me logged in", help="Stay logged in for 7 days")
        
        # Login button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            login_button = st.form_submit_button(
                "🚀 LOGIN TO DASHBOARD",
                use_container_width=True
            )
    
    # Process login
    if login_button:
        if not username or not password:
            st.error("❌ Please enter both username/email and password")
        else:
            with st.spinner("🔍 Authenticating farmer..."):
                # Authenticate farmer
                auth_result = st.session_state.auth_db.authenticate_farmer(
                    username, password, ip_address="127.0.0.1"
                )
                
                if auth_result["success"]:
                    # Create session
                    session_result = st.session_state.auth_db.create_session(
                        auth_result["farmer_id"],
                        ip_address="127.0.0.1",
                        user_agent="Streamlit Browser"
                    )
                    
                    if session_result["success"]:
                        # Store session in streamlit session state
                        st.session_state.authenticated = True
                        st.session_state.farmer_id = auth_result["farmer_id"]
                        st.session_state.username = auth_result["username"]
                        st.session_state.full_name = auth_result["full_name"]
                        st.session_state.session_id = session_result["session_id"]
                        st.session_state.session_token = session_result["session_token"]
                        
                        st.success(f"✅ Welcome back, {auth_result['full_name']}!")
                        
                        # Redirect message
                        st.info("🚀 Redirecting to your agricultural dashboard...")
                        st.markdown("""
                        <script>
                        setTimeout(() => {
                            window.location.reload();
                        }, 2000);
                        </script>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Session creation failed. Please try again.")
                else:
                    st.error(f"❌ {auth_result['message']}")
    
    # Additional options
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔑 Forgot Password?", help="Reset your password"):
            st.info("📧 Password reset functionality coming soon!")
    with col2:
        if st.button("❓ Need Help?", help="Get support"):
            st.info("📞 Contact: support@maharashtra-krushi.gov.in")

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