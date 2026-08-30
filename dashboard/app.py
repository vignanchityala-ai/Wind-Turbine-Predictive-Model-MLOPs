import streamlit as st
import os
from api_client import WindTurbineAPIClient

# Configure page settings
st.set_page_config(
    page_title="Wind Turbine MLOps",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS for premium styling
css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path, "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize API Client in session state
if "api" not in st.session_state:
    st.session_state.api = WindTurbineAPIClient()

st.title("🌀 Wind Turbine Predictive Maintenance")
st.markdown("### Early Fault Detection & Monitoring System")
st.markdown("Welcome to the Operations Dashboard. Navigate using the sidebar to analyze turbine health, perform batch predictions, or monitor model performance.")

# Health Check Dashboard
health = st.session_state.api.get_health()
st.markdown("---")
st.subheader("System Status")

col1, col2 = st.columns(2)
with col1:
    if health["status"] == "ok":
        st.metric(label="API Status", value="ONLINE", delta="connected", delta_color="normal")
    else:
        st.metric(label="API Status", value="OFFLINE", delta="unreachable", delta_color="inverse")

with col2:
    st.metric(label="Available Models", value=health["models_available"])

st.markdown("---")
st.info("👈 **Select a page from the sidebar to get started.**")
