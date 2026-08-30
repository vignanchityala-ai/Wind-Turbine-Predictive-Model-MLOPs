import streamlit as st
import os
import requests
from api_client import WindTurbineAPIClient
import plotly.graph_objects as go
import time

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style.css")
with open(css_path, "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "api" not in st.session_state:
    st.session_state.api = WindTurbineAPIClient()

st.header("📡 API & Drift Monitoring")
st.markdown("Real-time observability into the FastAPI backend (Powered by Prometheus).")

# Fetch metrics from /metrics
try:
    resp = requests.get(f"{st.session_state.api.base_url}/metrics", timeout=2)
    if resp.status_code == 200:
        metrics = resp.text
        
        # Super simple parsing for demo
        # A real dashboard would query Prometheus via PromQL
        req_count = 0
        for line in metrics.split('\n'):
            if line.startswith('http_requests_total{') and 'method="POST"' in line:
                try:
                    req_count += float(line.split(' ')[1])
                except:
                    pass
        
        st.subheader("System Health")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total POST Requests Served", int(req_count))
        with col2:
            st.metric("Prometheus Exporter", "ACTIVE", delta="live")
            
        st.markdown("---")
        st.subheader("Data Freshness Watchdog")
        st.info("The FastAPI backend enforces a 60-minute data freshness check. Any predictions made on older SCADA payloads will surface a `warning` attribute in the response.")
        
        # Simulating API latency over time
        st.subheader("Simulated API Latency (ms)")
        import numpy as np
        t = np.arange(100)
        latency = 45 + np.random.normal(0, 5, 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=latency, mode='lines', line=dict(color='#10b981')))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f0f4f8'),
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="ms"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error(f"Failed to fetch metrics: {resp.status_code}")
except Exception as e:
    st.error(f"Could not connect to /metrics: {e}")
