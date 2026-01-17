**Health endpoint & keep-alive instructions**

- **Files added/changed:** `health_server.py`, `app.py`, `requirements.txt`

- **Purpose:** Provide a lightweight `/health` HTTP endpoint so an external uptime monitor can ping the app and avoid idling/sleeping on hosts that support it.

- **How it works:** `health_server.py` runs a tiny Flask app exposing `/health` returning `{"status":"ok"}`. `app.py` attempts to start it in a background thread at startup. If the host forbids binding additional ports (Streamlit Cloud, some PaaS), `app.py` catches the error and continues.

- **Usage (self-hosted / VM / container):**

  1. Install requirements:

  ```bash
  pip install -r requirements.txt
  ```

  2. Run the Streamlit app (example):

  ```bash
  streamlit run app.py
  ```

  3. Start or ensure `health_server.py` is running (the app tries to start it automatically). The health endpoint defaults to port `8000`. To override, set environment variable `HEALTH_PORT`.

  4. Point your uptime monitor (UptimeRobot, Pingdom, etc.) to `http://<host>:8000/health` and configure checks every 5–15 minutes.

- **Hosting-specific notes:**
  - Streamlit Cloud: free-tier apps may still sleep; Streamlit Cloud may prevent binding a second port. Upgrade to a paid plan or host on a VM/container.
  - Azure App Service: set `Always On` = On in the App Service configuration.
  - Heroku/Railway/GCP App Engine: free tiers typically sleep; use paid tiers or provider health settings.
  - Windows/IIS: consider running Streamlit as a Windows Service or set the app pool `Idle Time-out` to 0 and `Start Mode` to `AlwaysRunning`.

- **Security:** Exposing a health endpoint is low-risk, but if you want to restrict it, run the health server only on an internal port and expose it through a reverse proxy, or protect it with a simple token check.
