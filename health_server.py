from threading import Thread
from flask import Flask, jsonify
import os

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify(status="ok"), 200


def run():
    port = int(os.getenv("HEALTH_PORT", "8000"))
    # Use 0.0.0.0 so the service is reachable from other machines/containers
    app.run(host="0.0.0.0", port=port)


def start_in_thread():
    """Start the health server in a background daemon thread.

    Useful when you want a lightweight HTTP health endpoint running
    alongside a Streamlit process on the same host/VM. Note: some
    managed hosts (Streamlit Cloud, certain PaaS) do not allow binding
    additional ports — in that case run this as a separate process or
    use the hosting provider's native health-check settings.
    """
    t = Thread(target=run, daemon=True)
    t.start()


if __name__ == "__main__":
    run()
