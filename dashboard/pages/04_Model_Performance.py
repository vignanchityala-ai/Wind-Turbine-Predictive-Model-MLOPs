import streamlit as st
import os
from api_client import WindTurbineAPIClient

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style.css")
with open(css_path, "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "api" not in st.session_state:
    st.session_state.api = WindTurbineAPIClient()

st.header("📊 Model Performance")
st.markdown("Metrics tracked from the MLflow model registry.")

farms_data = st.session_state.api.get_farms().get("farms", {})
if not farms_data:
    st.info("No farm configurations found.")
    st.stop()

selected_farm = st.selectbox("Select Farm Model", options=list(farms_data.keys()))
model_name = f"Wind_Farm_{selected_farm}_farm_model"
info = st.session_state.api.get_model_info(model_name)

if not info:
    st.warning("Model info not found.")
    st.stop()

st.subheader(f"Metrics for {selected_farm} Global Model")
col1, col2, col3 = st.columns(3)

with col1:
    care_score = info.get("care_composite")
    val = f"{care_score:.4f}" if care_score else "N/A"
    st.metric("CARE Composite Score", val)

with col2:
    st.metric("Model Version", info.get("model_version", "N/A"))

with col3:
    features = info.get("n_features", 0)
    st.metric("Feature Count", features)

st.markdown("---")
st.markdown("### Feature Schema Details")
st.write(f"**Schema Hash:** `{info.get('feature_schema_hash', 'N/A')}`")
st.write(f"**Training Strategy:** `{info.get('training_farms', ['N/A'])}`")
st.write(f"**Anomaly Threshold:** `{info.get('threshold', 'N/A')}`")

st.info("Note: For Isolation Forest models, per-feature attribution is intentionally omitted as it is generally unreliable for causal inferences.")
