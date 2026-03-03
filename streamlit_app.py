import json
import logging
import os
import time

import redis
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler

from frontend.analytics_tab import render_analytics_tab
from frontend.diagnosis_tab import render_diagnosis_tab, render_results
from frontend.history_tab import render_history_tab
from pneumonia_classifier.utils.auth import MOCK_USERS_DB, verify_password
from pneumonia_classifier.utils.database import init_db, purge_old_records

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Pneumonia AI Diagnostic Suite",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    :root {
        --primary-color: #00d2ff;
        --secondary-color: #3a7bd5;
        --bg-color: #0e1117;
    }
    .main {
        background-color: var(--bg-color);
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        border: 1px solid #60a5fa;
    }
    .status-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.875rem;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'role' not in st.session_state:
    st.session_state.role = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None


# --- LOGIN PAGE ---
def login_page():
    st.title("Diagnostic Portal Login")
    with st.container():
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown("### Secure Clinical Access")
            username = st.text_input("Physician ID", key="login_username")
            password = st.text_input("Passcode", type="password", key="login_password")
            if st.button("Authorize Access", key="login_btn"):
                user_dict = MOCK_USERS_DB.get(username)
                if user_dict and verify_password(password, user_dict["hashed_password"]):
                    st.session_state.authenticated = True
                    st.session_state.user = username
                    st.session_state.role = user_dict["role"]
                    st.rerun()
                else:
                    st.error("Invalid credentials. Audit log entry created.")

            st.markdown("""
            ---
            **Demo Credentials:**
            - **Nurse Access:** `nurse_joy` / `nurse123`
            - **Doctor Access:** `dr_smith` / `doc123`
            """)


# --- MAIN DASHBOARD ---
def dashboard():
    st.sidebar.title(f"Logged in as {st.session_state.user}")
    st.sidebar.info(f"Role: {st.session_state.role}")

    # --- SYSTEM OVERVIEW SIDEBAR ---
    st.sidebar.divider()
    st.sidebar.header("System Overview")
    st.sidebar.markdown("""
    **Architecture: Custom CNN (Net)**
    High-performance diagnostic model optimized for:
    1. **Texture Localization**
    2. **Consolidation Detection**
    3. **Clinical Interpretability**

    **Quick Guide:**
    1. **Upload** or select a **Sample X-ray**.
    2. Click **Initiate AI Analysis**.
    3. View **Grad-CAM Heatmap** to see where the AI focused.
    4. Download the **Clinical PDF** for your records.
    """)

    st.sidebar.divider()
    if st.sidebar.button("Logout", key="logout_btn"):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.rerun()


    tab1, tab2, tab3 = st.tabs(["New Diagnosis", "Patient Records", "Clinical Analytics"])

    with tab1:
        render_diagnosis_tab()

    with tab2:
        render_history_tab()

    with tab3:
        render_analytics_tab()


# --- REPORT PAGE ---
def render_report_page(job_id):
    st.title("AI Diagnostic Report")
    from pneumonia_classifier.config import config
    REDIS_URL = config.REDIS_URL

    try:
        redis_conn = redis.Redis.from_url(REDIS_URL, decode_responses=True)

        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        # Display static info while waiting
        st.info(f"Tracking Background Analysis for Job ID: `{job_id}`")

        while True:
            raw_data = redis_conn.get(job_id)
            if raw_data:
                result = json.loads(raw_data)
                if result["status"] == "completed":
                    status_placeholder.empty()
                    progress_bar.empty()

                    # Construct result dict for the renderer
                    result_data = {
                        "patient_id": result.get("patient_id", job_id.replace("job_", "P-")),
                        "prediction": result["prediction"],
                        "confidence": result["confidence"],
                        "heatmap_base64": result.get("heatmap", ""),
                        "original_image": result.get("original_image", "")
                    }
                    render_results(result_data)
                    return
                elif result["status"] == "failed":
                    status_placeholder.error(f"Inference failed: {result['message']}")
                    return
                else:
                    status_placeholder.markdown(f"**Current Status:** {result.get('message', 'Processing...')}")
                    progress_bar.progress(50) # Indeterminate-ish
            else:
                status_placeholder.warning("Waiting for job to be registered in state store...")

            time.sleep(0.5)
            st.rerun() # Necessary for Streamlit to Refresh state

    except Exception as e:
        st.error(f"Error accessing report store: {e}")

    if st.button("Return to Diagnostic Suite"):
        st.query_params.clear()
        st.rerun()

# --- IMAGE VIEWER ---

def image_viewer():
    params = st.query_params
    if "viewer" in params:
        img_path = params["viewer"]
        st.title("Diagnostic Heatmap Viewer")
        if os.path.exists(img_path):
            st.image(img_path, width='stretch')
            st.info(f"Viewing specialized focus region: {img_path}")
        else:
            st.error("Diagnostic image not found on server.")

        if st.button("Back to Diagnostic Suite"):
            st.query_params.clear()
            st.rerun()

# --- APP ENTRY ---
@st.cache_resource
def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(purge_old_records, 'interval', days=1, args=[30])
    scheduler.start()
    logger.info("Started Data Retention Scheduler (30-day purge).")
    return scheduler

@st.cache_resource
def ensure_db_initialized():
    init_db()
    return True

def main():
    ensure_db_initialized()
    st.session_state.scheduler = start_scheduler()

    # Check for direct image viewing request
    if "viewer" in st.query_params:
        image_viewer()
        return

    # Check for diagnostic report request
    if "job" in st.query_params:
        render_report_page(st.query_params["job"])
        return

    if not st.session_state.authenticated:
        login_page()
    else:
        dashboard()

if __name__ == "__main__":
    main()




