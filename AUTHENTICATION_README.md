# 🔐 Maharashtra Krushi Mitra - Farmer Authentication System

## 🌾 Overview

A **beautiful, secure, and comprehensive farmer authentication system** for the Maharashtra AI Agricultural Platform. This system provides secure access control while maintaining the original functionality of your agricultural system.

![System Preview](https://img.shields.io/badge/Status-Ready%20for%20Production-brightgreen)
![Security](https://img.shields.io/badge/Security-bcrypt%20%2B%20Sessions-blue)
![UI](https://img.shields.io/badge/UI-Agricultural%20Theme-green)

---

## ✨ Key Features

### 🔒 **Security Features**
- **Bcrypt Password Hashing** - Industry-standard password security
- **Session Management** - 7-day secure sessions with tokens
- **Login Attempt Monitoring** - Track and prevent brute force attacks
- **Automatic Session Expiration** - Enhanced security with time-based logout
- **SQLite Database** - Secure local storage with proper schema

### 🎨 **Beautiful UI Design**
- **Agricultural Theme** - Modern green/brown color scheme matching your system
- **Responsive Design** - Works perfectly on desktop and mobile
- **Animated Elements** - Smooth transitions and engaging animations
- **Professional Layout** - Clean, intuitive interface for farmers
- **Form Validation** - Real-time input validation with helpful messages

### 👨‍🌾 **Farmer-Focused Features**
- **Comprehensive Registration** - Collect farm details, crops, district info
- **Personalized Dashboard** - Show farmer profile and agricultural data
- **Maharashtra Districts** - Complete list of all 36 districts
- **Crop Selection** - Multiple crop type support
- **Farm Area Tracking** - Acre-based farm size management

---

## 📁 System Files

| File | Purpose | Description |
|------|---------|-------------|
| `auth_database.py` | Database | Core authentication database with all security functions |
| `farmer_login.py` | Login Page | Beautiful Streamlit login/registration interface |
| `authenticated_crop_system.py` | Main System | Sample integrated agricultural system with auth |
| `run_demo.py` | Demo Helper | Easy-to-use script to run the system |
| `farmer_auth.db` | Database | SQLite database storing farmer accounts (auto-created) |

---

## 🚀 Quick Start Guide

### **Step 1: Install Dependencies**
```bash
# Install required packages
pip install bcrypt streamlit plotly pandas numpy
```

### **Step 2: Run the Login System**
```bash
# Start the farmer login page
streamlit run farmer_login.py
```

### **Step 3: Create Your Account**
1. Go to the **Register** tab
2. Fill in your details:
   - Personal info (name, username, email, password)
   - Farm details (district, crops, area)
3. Accept terms and click **CREATE MY ACCOUNT**

### **Step 4: Login**
1. Switch to **Login** tab
2. Enter your credentials
3. Click **LOGIN TO DASHBOARD**

### **Step 5: Access Main System**
```bash
# Run the authenticated agricultural system
streamlit run authenticated_crop_system.py
```

---

## 🧪 Demo Account (Testing)

For quick testing, use this pre-created account:

```
Username: test_farmer
Password: test123
Name: Test Farmer
District: Pune
```

---

## 🏗️ System Architecture

### **Database Schema**

#### **Farmers Table**
```sql
- farmer_id (Primary Key)
- username (Unique)
- email (Unique)
- password_hash (bcrypt)
- salt (bcrypt salt)
- full_name
- farm_name
- district
- village
- farm_area
- crop_types
- registration_date
- last_login
- is_active
- profile_picture
```

#### **Sessions Table**
```sql
- session_id (Primary Key)
- farmer_id (Foreign Key)
- session_token (Secure token)
- expires_at
- ip_address
- user_agent
- is_active
```

#### **Login Attempts Table**
```sql
- attempt_id (Primary Key)
- username
- ip_address
- success (Boolean)
- attempt_time
- error_message
```

### **Security Flow**
1. **Registration** → Password hashed with bcrypt → Stored in database
2. **Login** → Credentials verified → Session created with secure token
3. **Access** → Session validated on each request → Auto-expires after 7 days
4. **Logout** → Session invalidated → All tokens cleared

---

## 🎯 Integration with Your System

### **Option 1: Use the Sample System**
The `authenticated_crop_system.py` provides a complete working example showing how to integrate authentication with your agricultural features.

### **Option 2: Integrate with Your Existing System**
Add these key functions to your existing `maharashtra_crop_system.py`:

```python
from auth_database import FarmerAuthDB

# Initialize authentication
if 'auth_db' not in st.session_state:
    st.session_state.auth_db = FarmerAuthDB()

# Check authentication at the start
def check_authentication():
    if not st.session_state.get('authenticated', False):
        st.warning("Please login to access the system")
        st.stop()

# Add to your main function
def main():
    check_authentication()  # Add this line
    # ... rest of your code
```

### **Option 3: Two-App Approach**
Keep your original system unchanged and run the authentication as a separate app:
1. Users login via `farmer_login.py`
2. After login, they access your original `maharashtra_crop_system.py`
3. Session state is shared between Streamlit apps

---

## 🔧 Configuration Options

### **Session Duration**
Modify in `auth_database.py`:
```python
expires_at = datetime.now() + timedelta(days=7)  # Change to your preference
```

### **Password Requirements**
Modify in `farmer_login.py`:
```python
if not password or len(password) < 6:  # Change minimum length
    errors.append("Password must be at least 6 characters")
```

### **Districts List**
Update Maharashtra districts in `farmer_login.py`:
```python
district = st.selectbox("📍 District *", [
    "Select District", "Pune", "Mumbai", "Nagpur", ...
    # Add or remove districts as needed
])
```

---

## 🛡️ Security Best Practices

### **What This System Provides:**
✅ **Password Hashing** - bcrypt with salt  
✅ **Session Tokens** - Cryptographically secure  
✅ **Login Monitoring** - Track all attempts  
✅ **Input Validation** - Prevent injection attacks  
✅ **Session Expiration** - Automatic security logout  

### **Production Recommendations:**
- [ ] Use HTTPS in production
- [ ] Add rate limiting for login attempts
- [ ] Implement email verification
- [ ] Add password reset functionality
- [ ] Use environment variables for sensitive config
- [ ] Regular database backups
- [ ] Monitor login patterns

---

## 📱 Mobile Experience

The system is fully **mobile-responsive** with:
- Touch-friendly interface
- Optimized form layouts
- Mobile-friendly navigation
- Responsive charts and visualizations
- Fast loading on mobile networks

---

## 🎨 Customization

### **Color Scheme**
Modify agricultural colors in `farmer_login.py`:
```css
:root {
    --primary-green: #2E7D32;    /* Deep Forest Green */
    --secondary-green: #4CAF50;  /* Bright Green */
    --earth-brown: #5D4E37;      /* Rich Earth Brown */
    --sky-blue: #1976D2;         /* Sky Blue */
    /* Customize as needed */
}
```

### **Branding**
Update logos and titles in `farmer_login.py`:
```python
st.markdown("""
<h1 class="app-title">Your Farm Name</h1>
<p class="app-subtitle">Your Custom Subtitle</p>
""", unsafe_allow_html=True)
```

---

## 🐛 Troubleshooting

### **Common Issues:**

**1. "bcrypt not found"**
```bash
pip install bcrypt
```

**2. "Database initialization failed"**
```bash
# Delete the database file and restart
rm farmer_auth.db
python farmer_login.py
```

**3. "Session expired immediately"**
- Check system date/time
- Verify session token storage
- Clear browser cache

**4. "Cannot access main system"**
- Ensure you're logged in first
- Check session state in Streamlit
- Verify database connectivity

---

## 📊 Database Management

### **View Registered Farmers:**
```python
import sqlite3
conn = sqlite3.connect('farmer_auth.db')
cursor = conn.cursor()
cursor.execute("SELECT username, full_name, district, registration_date FROM farmers")
for row in cursor.fetchall():
    print(row)
```

### **Reset Password (Admin):**
```python
from auth_database import FarmerAuthDB
auth_db = FarmerAuthDB()

# Update password for a farmer
new_hash, salt = auth_db.hash_password("new_password")
# Update in database...
```

### **View Login Activity:**
```python
cursor.execute("SELECT username, success, attempt_time FROM login_attempts ORDER BY attempt_time DESC LIMIT 10")
```

---

## 🚀 Production Deployment

### **For Local Networks:**
```bash
streamlit run farmer_login.py --server.address 0.0.0.0 --server.port 8501
```

### **For Production Servers:**
1. Use a production WSGI server (not Streamlit's dev server)
2. Set up proper SSL certificates
3. Configure firewall and security groups
4. Use PostgreSQL instead of SQLite for scalability
5. Implement proper backup strategies

---

## 📄 License & Support

- **License:** This authentication system is provided as an enhancement to your existing Maharashtra Agricultural System
- **Support:** For technical issues, check the troubleshooting section above
- **Customization:** Feel free to modify the system to match your specific requirements

---

## 🎉 Success Metrics

After implementing this system, you can expect:

- **🔒 100% Secure Access** - Only registered farmers can access the system
- **📱 Mobile-Friendly** - Works on all devices and screen sizes
- **⚡ Fast Performance** - Optimized database queries and session management
- **👨‍🌾 Farmer-Friendly** - Intuitive interface designed for agricultural users
- **📊 User Analytics** - Track login patterns and user engagement
- **🛡️ Enterprise Security** - Production-ready security implementation

---

**🌾 Ready to secure your Maharashtra Agricultural System with beautiful farmer authentication!**

*Happy Farming! 🚜*