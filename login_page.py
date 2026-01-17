import streamlit as st
from streamlit_option_menu import option_menu
import re
import time

# This module provides `render_login()` and `render_register()` and a top-level
# `render()` that shows the selected tab. It does NOT execute the UI on import
# so it can be integrated into `app.py`.

# Initialize session state for toggles if not present
if "show_password" not in st.session_state:
    st.session_state["show_password"] = False
if "show_reg_pass" not in st.session_state:
    st.session_state["show_reg_pass"] = False
if "show_reg_confirm" not in st.session_state:
    st.session_state["show_reg_confirm"] = False
if "light_mode" not in st.session_state:
    st.session_state["light_mode"] = False


def _get_css():
    """Return the CSS string used by the login page (uses session light/dark)."""
    dark_vars = {
        "bg_1": "#0f172a",
        "bg_2": "#102235",
        "accent": "#16a34a",
        "accent_dark": "#0f8a3d",
        "glass_bg": "rgba(255,255,255,0.08)",
        "glass_border": "rgba(255,255,255,0.18)",
        "text": "#e6eef0",
    }
    light_vars = {
        "bg_1": "#f6faf6",
        "bg_2": "#ecf7ea",
        "accent": "#0ea54b",
        "accent_dark": "#0a7a34",
        "glass_bg": "rgba(255,255,255,0.70)",
        "glass_border": "rgba(10,10,10,0.06)",
        "text": "#07241a",
    }

    vars = light_vars if st.session_state["light_mode"] else dark_vars

    CSS = f"""
    <style>
    :root {{
        --bg-1: {vars['bg_1']};
        --bg-2: {vars['bg_2']};
        --accent: {vars['accent']};
        --accent-dark: {vars['accent_dark']};
        --glass-bg: {vars['glass_bg']};
        --glass-border: {vars['glass_border']};
        --text: {vars['text']};
        --radius: 16px;
        --card-shadow: 0 10px 40px rgba(4,8,15,0.55);
    }}
    """

    # We include the full CSS from the user's design; to keep the patch concise,
    # we append the exact CSS content originally provided in the workspace.
    # For correctness the CSS will be re-inserted here at runtime.
    # (We'll assemble it by returning the large string below.)

    large_css = """
    /* Basic page */
    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(255,255,255,0.02), transparent 8%),
                    linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 60%);
        min-height: 100vh;
        color: var(--text);
        font-family: "Segoe UI", Roboto, -apple-system, sans-serif;
    }
    .bg-move { position: fixed; inset: 0; z-index: 0; pointer-events: none; background: linear-gradient(120deg, rgba(255,255,255,0.02), rgba(0,0,0,0.02)); animation: moveGradient 10s linear infinite; mix-blend-mode: overlay; }
    @keyframes moveGradient { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    .bg-dots { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
    .bg-dots span { position: absolute; display: block; width: 10px; height: 10px; border-radius: 50%; background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02)); opacity: 0.12; animation: floatY 8s ease-in-out infinite; }
    @keyframes floatY { 0% { transform: translateY(0) scale(1); opacity: 0.12; } 50% { transform: translateY(-40px) scale(1.2); opacity: 0.18; } 100% { transform: translateY(0) scale(1); opacity: 0.12; } }
    .center-wrap { position: relative; z-index: 5; display: flex; justify-content: center; align-items: center; padding: 36px 20px 80px 20px; }
    .floating-logo { position: absolute; top: 28px; left: 28px; display: flex; align-items: center; gap: 10px; z-index: 50; }
    .logo-bubble { width: 52px; height: 52px; border-radius: 12px; background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(0,0,0,0.06)); border: 1px solid var(--glass-border); display:flex; align-items:center; justify-content:center; backdrop-filter: blur(8px); animation: floatLogo 6s ease-in-out infinite; box-shadow: 0 6px 18px rgba(0,0,0,0.28); }
    @keyframes floatLogo { 0% { transform: translateY(0); } 50% { transform: translateY(-8px) rotate(-2deg); } 100% { transform: translateY(0); transform: rotate(0deg); } }
    .logo-text { font-weight: 700; color: var(--text); letter-spacing: 0.2px; }
    .login-card { width: 100%; max-width: 520px; padding: 34px; border-radius: var(--radius); background: var(--glass-bg); border: 1px solid var(--glass-border); backdrop-filter: blur(14px) saturate(120%); box-shadow: var(--card-shadow); position: relative; overflow: hidden; transition: transform 0.35s ease, box-shadow 0.35s ease; }
    .login-card:hover { transform: translateY(-6px); box-shadow: 0 18px 50px rgba(4,8,15,0.65); }
    .card-heading { text-align: center; margin-bottom: 10px; }
    .card-heading h1 { margin: 0; font-size: 26px; color: var(--text); }
    .card-heading p { margin: 6px 0 18px 0; color: rgba(255,255,255,0.75); font-size: 13px; }
    .stTextInput > label { color: rgba(255,255,255,0.85); }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.06) !important; color: var(--text) !important; padding: 12px 14px !important; border-radius: 10px !important; outline: none !important; transition: box-shadow 0.2s ease, border-color 0.2s ease; height: 44px !important; }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { box-shadow: 0 6px 20px rgba(22,163,74,0.12) !important; border-color: var(--accent) !important; }
    .stCheckbox > label span { color: var(--text) !important; }
    button[kind="primary"] { background: linear-gradient(135deg, var(--accent), var(--accent-dark)) !important; border: none !important; color: white !important; padding: 12px 18px !important; border-radius: 12px !important; box-shadow: 0 8px 24px rgba(16,120,63,0.18) !important; transition: transform 0.18s ease, box-shadow 0.18s ease; font-weight: 700 !important; }
    button[kind="primary"]:hover { transform: translateY(-3px) !important; box-shadow: 0 22px 48px rgba(16,120,63,0.24) !important; }
    button[kind="secondary"] { background: transparent !important; border: 1px solid rgba(255,255,255,0.06) !important; color: var(--text) !important; }
    .helper-links { display:flex; justify-content:space-between; align-items:center; gap: 12px; margin-top: 8px; }
    .quick-grid { display:grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 16px; }
    .quick-card { background: rgba(255,255,255,0.02); border-radius: 10px; padding: 10px; font-size: 13px; }
    @media (max-width: 720px) { .center-wrap { padding-top: 72px; } .floating-logo { left: 18px; top: 12px; } .login-card { padding: 22px; border-radius: 14px; } .card-heading h1 { font-size: 20px; } .quick-grid { grid-template-columns: 1fr; } }
    #MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
    </style>
    """

    return CSS + large_css


def _render_background():
    bg_html = """
    <div class="bg-move"></div>
    <div class="bg-dots">
        <span style="left:5%; top:20%; width:8px; height:8px; animation-duration:7s;"></span>
        <span style="left:18%; top:60%; width:10px; height:10px; animation-duration:9s;"></span>
        <span style="left:35%; top:15%; width:12px; height:12px; animation-duration:11s;"></span>
        <span style="left:50%; top:65%; width:9px; height:9px; animation-duration:8s;"></span>
        <span style="left:68%; top:28%; width:6px; height:6px; animation-duration:10s;"></span>
        <span style="left:82%; top:72%; width:11px; height:11px; animation-duration:12s;"></span>
    </div>
    """
    st.markdown(bg_html, unsafe_allow_html=True)


def _render_logo():
    vars = {
        'accent': '#16a34a',
        'accent_dark': '#0f8a3d',
        'text': '#e6eef0'
    }
    logo_html = f"""
    <div class="floating-logo">
      <div class="logo-bubble" title="MahaAgroAI">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M4 20c4-6 10-9 16-9" stroke="{vars['accent']}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M12 3c1.6 1.8 3 4 4 6 2 4 4 7 6 9" stroke="{vars['accent_dark']}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="logo-text">MahaAgroAI</div>
    </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)


# small helper used in UI
def _show_loading_then_success():
    with st.spinner("Authenticating..."):
        time.sleep(1.0)
    st.success("✅ Login successful!")
    time.sleep(0.7)
    st.info("🌾 Loading your agricultural dashboard...")


# Public render functions

def render_login(auth_hook=None):
    """Render the login UI.

    Parameters:
    - auth_hook: optional callable(username, password) -> bool. If provided,
      it's used to validate credentials. If not provided, a demo check is used
      (username=="demo" and password=="demo").

    Returns True if login succeeded (session state will be updated), False otherwise.
    """
    # Inject CSS and background
    st.markdown(_get_css(), unsafe_allow_html=True)
    _render_background()
    _render_logo()

    selected = option_menu(
        menu_title=None,
        options=["Login"],
        icons=["lock-fill"],
        orientation="horizontal",
        styles={
            "container": {"justify-content": "center", "display": "flex", "gap": "14px"},
            "nav-link": {"font-size": "16px", "color": "var(--text)"},
            "icon": {"color": "var(--accent)"},
            "nav-link-selected": {"background-color": "rgba(255,255,255,0.03);", "color": "var(--accent)", "border-radius": "10px"},
        }
    )

    st.markdown('<div class="center-wrap">', unsafe_allow_html=True)

    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card-heading">
            <h1>Welcome Back</h1>
            <p>Secure access to MahaAgroAI — farmer dashboard</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    login_success = False
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username or Email", placeholder="Enter your username or email", help="Use the email or username you registered with")
        col_pwd, col_eye = st.columns([4, 1], gap="small")
        with col_pwd:
            password = st.text_input("Password",
                                     type="password" if not st.session_state.get('show_password', False) else "text",
                                     placeholder="Enter your password",
                                     help="Enter your secure password")
        with col_eye:
            st.checkbox("👁️", key="show_password", help="Show/hide password")

        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            remember = st.checkbox("Keep me signed in", help="Stay logged in for 7 days")
        with col2:
            pass

        submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")

    if submit:
        if not username or not password:
            st.error("Please enter both username and password")
        else:
            # call provided auth hook if available, otherwise demo check
            try:
                if auth_hook is not None:
                    ok = auth_hook(username.strip(), password)
                else:
                    ok = (username.strip().lower() == "demo" and password == "demo")
            except Exception as e:
                ok = False
                st.error(f"Authentication error: {e}")

            if ok:
                _show_loading_then_success()
                st.session_state['authenticated'] = True
                st.session_state['user_info'] = {'name': username}
                login_success = True
                # allow app to react to login
            else:
                st.error("Invalid username or password")

    # helper links and quick access cards
    st.markdown(
        """
        <div class="helper-links">
            <div style="display:flex; gap:10px;">
                <button kind="secondary" class="stButton" style="border-radius:10px; padding:6px 10px;">🔑 Reset Password</button>
                <a href="mailto:support@maharashtra-krushi.gov.in" style="color:inherit; text-decoration:none; margin-left:8px;">❓ Need Help?</a>
            </div>
            <div style="font-size:13px; color: rgba(255,255,255,0.65);">Safe • Secure • AI-assisted</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="quick-grid">
            <div class="quick-card">🌱 <strong>Disease Detection</strong><div style="opacity:0.7; font-size:12px;">Scan a leaf and get quick diagnosis</div></div>
            <div class="quick-card">🌦️ <strong>Weather Alert</strong><div style="opacity:0.7; font-size:12px;">Personalized weather notifications</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    return login_success


def render_register():
    st.markdown(_get_css(), unsafe_allow_html=True)
    _render_background()
    _render_logo()

    option_menu(
        menu_title=None,
        options=["Register"],
        icons=["person-plus-fill"],
        orientation="horizontal",
    )

    st.markdown('<div class="center-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card-heading">
            <h1>Create Account</h1>
            <p>Join the MahaAgroAI farming community</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("registration_form", clear_on_submit=True):
        col1, col2 = st.columns(2, gap="small")
        with col1:
            full_name = st.text_input("Full Name", placeholder="Enter your full name")
            phone = st.text_input("Phone Number", placeholder="10-digit mobile number")
        with col2:
            email = st.text_input("Email", placeholder="your.email@example.com")
            district = st.selectbox("District", options=["Select District", "Pune", "Mumbai", "Nagpur", "Nashik", "Aurangabad", "Solapur", "Ahmednagar", "Kolhapur"])

        st.markdown('<div style="margin:8px 0 6px;"><strong>Create Password</strong></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="small")
        with col1:
            pass_col, toggle_col = st.columns([4, 1], gap="small")
            with pass_col:
                password = st.text_input("Password", type="password" if not st.session_state.get('show_reg_pass', False) else "text", placeholder="Create strong password")
            with toggle_col:
                st.checkbox("👁️", key="show_reg_pass", help="Show/hide password")
        with col2:
            confirm_col, confirm_toggle = st.columns([4, 1], gap="small")
            with confirm_col:
                confirm_password = st.text_input("Confirm Password", type="password" if not st.session_state.get('show_reg_confirm', False) else "text", placeholder="Confirm password")
            with confirm_toggle:
                st.checkbox("👁️", key="show_reg_confirm", help="Show/hide password")

        col1, col2 = st.columns([1, 1], gap="small")
        with col1:
            terms = st.checkbox("I accept the Terms of Service")
        with col2:
            newsletter = st.checkbox("Send me farming tips", value=True)

        submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")

    if submit:
        if not all([full_name, email, phone, password, confirm_password]):
            st.error("Please fill all required fields")
            return
        if not terms:
            st.error("Please accept the Terms of Service")
            return
        if password != confirm_password:
            st.error("Passwords do not match")
            return
        if not re.match(r"^[0-9]{10}$", phone.strip()):
            st.error("Please enter a valid 10-digit phone number")
            return
        if len(password) < 8:
            st.error("Password must be at least 8 characters long")
            return
        if district == "Select District":
            st.error("Please select your district")
            return

        with st.spinner("Creating your account..."):
            time.sleep(1.0)
        st.success("✅ Account created successfully!")
        st.info("📧 Please check your email for verification link")

        st.markdown(
            f"""
            <div style="margin-top:12px; border-radius:10px; background: rgba(255,255,255,0.02); padding:12px;">
                <div style="font-weight:700; margin-bottom:6px;">Welcome to MahaAgroAI, {full_name} 🎉</div>
                <div style="opacity:0.85; font-size:13px;">
                    Your account has been created. You can now login to access:
                    <ul style="margin:8px 0 0 1.1rem;">
                        <li>AI-powered crop analysis</li>
                        <li>Weather monitoring & alerts</li>
                        <li>Personalized farming advice</li>
                        <li>Community support</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# convenience wrapper: if other modules import this and want the default
# combined render behavior, call render().
def render(mode='login', auth_hook=None):
    """Render the login or register UI.

    - mode: 'login' or 'register' (defaults to 'login')
    - auth_hook: optional callable(username,password) -> bool for authentication
    """
    if mode == 'register':
        return render_register()
    else:
        return render_login(auth_hook=auth_hook)


if __name__ == "__main__":
    # When run directly, show login (demo mode)
    render()

# Helper: small spinner + success animation after login
def show_loading_then_success():
    with st.spinner("Authenticating..."):
        time.sleep(1.0)
    st.success("✅ Login successful!")
    time.sleep(0.7)
    st.info("🌾 Loading your agricultural dashboard...")

# Login tab
def login_tab():
    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card-heading">
            <h1>Welcome Back</h1>
            <p>Secure access to MahaAgroAI — farmer dashboard</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username or Email", placeholder="Enter your username or email", help="Use the email or username you registered with")
        col_pwd, col_eye = st.columns([4, 1], gap="small")
        with col_pwd:
            password = st.text_input("Password",
                                     type="password" if not st.session_state.get('show_password', False) else "text",
                                     placeholder="Enter your password",
                                     help="Enter your secure password")
        with col_eye:
            st.checkbox("👁️", key="show_password", help="Show/hide password")

        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            remember = st.checkbox("Keep me signed in", help="Stay logged in for 7 days")
        with col2:
            pass

        submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")

    if submit:
        if not username or not password:
            st.error("Please enter both username and password")
        else:
            # demo auth
            if username.strip().lower() == "demo" and password == "demo":
                show_loading_then_success()
                # Replace this with actual redirect/logic in your app
                st.balloons()
            else:
                st.error("Invalid username or password")

    # helper links and quick access cards
    st.markdown(
        """
        <div class="helper-links">
            <div style="display:flex; gap:10px;">
                <button kind="secondary" class="stButton" style="border-radius:10px; padding:6px 10px;">🔑 Reset Password</button>
                <a href="mailto:support@maharashtra-krushi.gov.in" style="color:inherit; text-decoration:none; margin-left:8px;">❓ Need Help?</a>
            </div>
            <div style="font-size:13px; color: rgba(255,255,255,0.65);">Safe • Secure • AI-assisted</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="quick-grid">
            <div class="quick-card">🌱 <strong>Disease Detection</strong><div style="opacity:0.7; font-size:12px;">Scan a leaf and get quick diagnosis</div></div>
            <div class="quick-card">🌦️ <strong>Weather Alert</strong><div style="opacity:0.7; font-size:12px;">Personalized weather notifications</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


# Register tab
def register_tab():
    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card-heading">
            <h1>Create Account</h1>
            <p>Join the MahaAgroAI farming community</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("registration_form", clear_on_submit=True):
        col1, col2 = st.columns(2, gap="small")
        with col1:
            full_name = st.text_input("Full Name", placeholder="Enter your full name")
            phone = st.text_input("Phone Number", placeholder="10-digit mobile number")
        with col2:
            email = st.text_input("Email", placeholder="your.email@example.com")
            district = st.selectbox("District", options=["Select District", "Pune", "Mumbai", "Nagpur", "Nashik", "Aurangabad", "Solapur", "Ahmednagar", "Kolhapur"])

        st.markdown('<div style="margin:8px 0 6px;"><strong>Create Password</strong></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="small")
        with col1:
            # nested columns to place toggle nicely on small screens
            pass_col, toggle_col = st.columns([4, 1], gap="small")
            with pass_col:
                password = st.text_input("Password", type="password" if not st.session_state.get('show_reg_pass', False) else "text", placeholder="Create strong password")
            with toggle_col:
                st.checkbox("👁️", key="show_reg_pass", help="Show/hide password")
        with col2:
            confirm_col, confirm_toggle = st.columns([4, 1], gap="small")
            with confirm_col:
                confirm_password = st.text_input("Confirm Password", type="password" if not st.session_state.get('show_reg_confirm', False) else "text", placeholder="Confirm password")
            with confirm_toggle:
                st.checkbox("👁️", key="show_reg_confirm", help="Show/hide password")

        st.markdown("", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1], gap="small")
        with col1:
            terms = st.checkbox("I accept the Terms of Service")
        with col2:
            newsletter = st.checkbox("Send me farming tips", value=True)

        submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")

    if submit:
        # validations
        if not all([full_name, email, phone, password, confirm_password]):
            st.error("Please fill all required fields")
            return
        if not terms:
            st.error("Please accept the Terms of Service")
            return
        if password != confirm_password:
            st.error("Passwords do not match")
            return
        if not re.match(r"^[0-9]{10}$", phone.strip()):
            st.error("Please enter a valid 10-digit phone number")
            return
        if len(password) < 8:
            st.error("Password must be at least 8 characters long")
            return
        if district == "Select District":
            st.error("Please select your district")
            return

        with st.spinner("Creating your account..."):
            time.sleep(1.0)
        st.success("✅ Account created successfully!")
        st.info("📧 Please check your email for verification link")

        # welcome card (HTML inside Streamlit)
        st.markdown(
            f"""
            <div style="margin-top:12px; border-radius:10px; background: rgba(255,255,255,0.02); padding:12px;">
                <div style="font-weight:700; margin-bottom:6px;">Welcome to MahaAgroAI, {full_name} 🎉</div>
                <div style="opacity:0.85; font-size:13px;">
                    Your account has been created. You can now login to access:
                    <ul style="margin:8px 0 0 1.1rem;">
                        <li>AI-powered crop analysis</li>
                        <li>Weather monitoring & alerts</li>
                        <li>Personalized farming advice</li>
                        <li>Community support</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # registration tips
    st.markdown(
        """
        <div style="margin-top:10px; display:grid; gap:8px;">
            <div style="font-size:13px; opacity:0.85;">Registration Tips</div>
            <div style="display:flex; gap:8px; flex-direction:column;">
                <div style="background: rgba(255,255,255,0.02); padding:8px; border-radius:8px;">✅ Use a valid email you can access</div>
                <div style="background: rgba(255,255,255,0.02); padding:8px; border-radius:8px;">🔒 Choose a strong password (mix letters, numbers)</div>
                <div style="background: rgba(255,255,255,0.02); padding:8px; border-radius:8px;">📱 Ensure your phone number is correct for alerts</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


# Render the correct tab
if selected == "Login":
    login_tab()
else:
    register_tab()

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div style='text-align:center; color: rgba(255,255,255,0.45); margin-top:24px; font-size:12px;'>
        Tip: On small screens the layout adapts — cards become more compact and action buttons expand to full width.
    </div>
    """,
    unsafe_allow_html=True
)
