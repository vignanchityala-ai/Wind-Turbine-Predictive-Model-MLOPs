import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style.css")
with open(css_path, "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.header("🔍 Turbine Deep-Dive")
st.markdown("Upload SCADA data to inspect the power curve and raw sensor time series visually.")

uploaded_file = st.file_uploader("Upload SCADA CSV for deep-dive", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Try to identify time column
    time_cols = [c for c in df.columns if "time" in c.lower()]
    time_col = time_cols[0] if time_cols else df.columns[0]
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Power Curve")
        # Try to identify wind and power
        wind_cols = [c for c in df.columns if "wind_speed" in c.lower() or "ws" in c.lower()]
        power_cols = [c for c in df.columns if "power" in c.lower() or "kw" in c.lower()]
        
        if wind_cols and power_cols:
            fig_pc = px.scatter(
                df, x=wind_cols[0], y=power_cols[0],
                opacity=0.5,
                color_discrete_sequence=['#06b6d4']
            )
            fig_pc.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f0f4f8'),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_pc, use_container_width=True)
        else:
            st.info("Could not identify wind speed and power columns automatically.")

    with col2:
        st.subheader("📈 Sensor Time Series")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        
        # Remove expected non-sensor columns
        for drop in ["asset_id", "status_type_id", "is_test"]:
            if drop in numeric_cols:
                numeric_cols.remove(drop)
                
        if numeric_cols:
            selected_sensor = st.selectbox("Select Sensor", options=numeric_cols)
            fig_ts = px.line(
                df, x=time_col, y=selected_sensor,
                color_discrete_sequence=['#d946ef']
            )
            fig_ts.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f0f4f8'),
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified"
            )
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("No numeric sensors found to plot.")
