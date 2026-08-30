import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from api_client import WindTurbineAPIClient

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style.css")
with open(css_path, "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "api" not in st.session_state:
    st.session_state.api = WindTurbineAPIClient()

st.header("📂 Upload & Batch Predict")
st.markdown("Upload a raw CSV dataset to evaluate it against a trained turbine model.")

# Fetch available models
health = st.session_state.api.get_health()
if health["status"] != "ok":
    st.error("API is offline. Cannot fetch models.")
    st.stop()

# We need a dedicated endpoint to get all models, or we can use /models. Wait, /models exists!
import requests
models_resp = requests.get(f"{st.session_state.api.base_url}/models")
model_names = models_resp.json().get("models", []) if models_resp.status_code == 200 else []

if not model_names:
    st.warning("No trained models found on the server.")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    selected_model = st.selectbox("Select Target Model", options=model_names)

with col2:
    uploaded_file = st.file_uploader("Upload SCADA CSV", type=["csv"])

if uploaded_file and st.button("Run Batch Prediction"):
    with st.spinner("Analyzing data through FastAPI..."):
        file_bytes = uploaded_file.getvalue()
        resp = st.session_state.api.batch_predict(selected_model, file_bytes)
        
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Prediction complete! Analyzed {data['n_readings']} readings.")
            
            threshold = data["threshold"]
            events = data["events"]
            
            st.markdown("### Detection Results")
            if events:
                st.markdown(f"<div class='status-anomaly'>🔴 ANOMALY DETECTED ({len(events)} Events)</div>", unsafe_allow_html=True)
                for e in events:
                    with st.expander(f"Event #{e['event_id']} ({e['duration_hours']:.1f} hours)"):
                        st.write(f"**Start:** {e['start']}")
                        st.write(f"**End:** {e['end']}")
                        st.write(f"**Peak Score:** {e['peak_score']:.4f} | **Mean Score:** {e['mean_score']:.4f}")
            else:
                st.markdown("<div class='status-normal'>🟢 NORMAL (No anomalies detected)</div>", unsafe_allow_html=True)
            
            # Plot the scores
            st.markdown("### Anomaly Score Timeline")
            scores = data["per_row_scores"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=scores, mode='lines', name='Anomaly Score', line=dict(color='#06b6d4')))
            fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444", annotation_text="Threshold")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f0f4f8'),
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")
