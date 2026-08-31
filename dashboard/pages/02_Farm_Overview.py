import streamlit as st
import os
from api_client import WindTurbineAPIClient

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style.css")
with open(css_path, "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "api" not in st.session_state:
    st.session_state.api = WindTurbineAPIClient()

st.header("🏢 Farm Overview")
st.markdown("High-level overview of available wind farms and their model readiness.")

farms_data = st.session_state.api.get_farms().get("farms", {})

if not farms_data:
    st.info("No farm configurations found or API unreachable.")
    st.stop()

# Display each farm in a grid
cols = st.columns(3)
for i, (farm_name, info) in enumerate(farms_data.items()):
    with cols[i % 3]:
        st.markdown(f"### {farm_name}")
        has_model = info.get("has_model", False)
        if has_model:
            st.markdown("<div class='status-normal'>🟢 Global Model Ready</div>", unsafe_allow_html=True)
            
            # Fetch model info
            model_name = f"Wind_Farm_{farm_name}_farm_model"
            m_info = st.session_state.api.get_model_info(model_name)
            
            st.write(f"**Version:** {m_info.get('model_version', 'N/A')}")
            st.write(f"**Trained:** {m_info.get('training_date', 'N/A')[:10]}")
            st.write(f"**Features:** {m_info.get('n_features', 0)}")
            
            if st.button("Run Farm Prediction", key=farm_name):
                st.info("Use the Upload page and select this farm's model to run a batch prediction.")
                
        else:
            st.markdown("<div class='status-anomaly'>🔴 No Model Available</div>", unsafe_allow_html=True)
        st.markdown("---")
