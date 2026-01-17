"""
Global authentication state management for Maharashtra Agricultural System
"""

import streamlit as st
from mongodb_auth import MongoFarmerAuth

def init_auth_state():
    """Initialize authentication state in session"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None

def check_auth():
    """Check if user is authenticated"""
    init_auth_state()
    return st.session_state.authenticated

def get_user_info():
    """Get authenticated user information"""
    return st.session_state.user_info if st.session_state.authenticated else None

def logout():
    """Clear authentication state"""
    st.session_state.authenticated = False
    st.session_state.user_info = None