#!/usr/bin/env python3
"""
Maharashtra Krushi Mitra - Demo Script
Shows how to run the authenticated agricultural system
"""

import os
import subprocess
import sys
from datetime import datetime

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🌾 MAHARASHTRA KRUSHI MITRA - AI AGRICULTURAL SYSTEM 🌾")
    print("=" * 60)
    print("Advanced AI-Powered Agricultural System with Farmer Authentication")
    print("© 2025 Maharashtra Agricultural Department")
    print("=" * 60)
    print()

def print_instructions():
    """Print usage instructions"""
    print("📋 SYSTEM OVERVIEW:")
    print("-------------------")
    print("✅ Secure farmer authentication system")
    print("✅ Beautiful agricultural-themed login page")
    print("✅ Comprehensive crop health analysis")
    print("✅ Weather and soil monitoring")
    print("✅ Pest risk assessment")
    print("✅ Personalized farmer dashboard")
    print("✅ Session management and security")
    print()
    
    print("🚀 HOW TO USE:")
    print("---------------")
    print("1. First, run the LOGIN PAGE to create account or login:")
    print("   Command: streamlit run farmer_login.py")
    print()
    print("2. After login, run the MAIN SYSTEM:")
    print("   Command: streamlit run authenticated_crop_system.py")
    print()
    print("   OR use your original system with authentication:")
    print("   Command: streamlit run maharashtra_crop_system.py")
    print()
    
    print("💡 DEMO ACCOUNT:")
    print("----------------")
    print("Username: test_farmer")
    print("Password: test123")
    print("(Already created for testing)")
    print()
    
    print("🔐 SECURITY FEATURES:")
    print("---------------------")
    print("• Password hashing with bcrypt")
    print("• Session management with tokens")
    print("• Login attempt monitoring")
    print("• Automatic session expiration")
    print("• Secure database storage")
    print()

def run_login_system():
    """Run the farmer login system"""
    print("🚀 Starting Farmer Login System...")
    print("Opening in your default browser...")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "farmer_login.py"])
    except KeyboardInterrupt:
        print("\n👋 Login system stopped!")
    except Exception as e:
        print(f"❌ Error running login system: {e}")

def run_main_system():
    """Run the main agricultural system"""
    print("🌾 Starting Main Agricultural System...")
    print("Make sure you're logged in first!")
    print("Opening in your default browser...")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "authenticated_crop_system.py"])
    except KeyboardInterrupt:
        print("\n👋 Agricultural system stopped!")
    except Exception as e:
        print(f"❌ Error running agricultural system: {e}")

def main():
    """Main demo function"""
    print_banner()
    print_instructions()
    
    while True:
        print("🎯 CHOOSE AN OPTION:")
        print("1. 🚪 Start Login Page (farmer_login.py)")
        print("2. 🌾 Start Main Agricultural System (authenticated_crop_system.py)")
        print("3. 📖 Show Instructions Again")
        print("4. ❌ Exit")
        print()
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            print()
            run_login_system()
            print()
        elif choice == "2":
            print()
            run_main_system()
            print()
        elif choice == "3":
            print()
            print_instructions()
        elif choice == "4":
            print("👋 Thank you for using Maharashtra Krushi Mitra!")
            print("🌱 Happy Farming! 🌾")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")
            print()

if __name__ == "__main__":
    main()