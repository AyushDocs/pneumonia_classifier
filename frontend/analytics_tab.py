import pandas as pd
import streamlit as st

from pneumonia_classifier.utils.database import get_all_predictions, get_drift_metrics


def render_analytics_tab():
    st.header("Clinical Intelligence Dashboard")

    with st.expander("Understanding Intelligence Metrics"):
        st.write("""
        - **Diagnostic Volume**: Tracks the daily throughput of the system.
        - **Confidence Distribution**: Monitors the stability of AI predictions. Narrowing distributions around 90%+ indicate high reliability.
        - **Model Drift (System Health)**: Compares current input statistics against the training baseline. Spikes here may indicate changes in imaging equipment or demographic shift.
        """)
    pred_data = get_all_predictions()
    drift_data = get_drift_metrics()

    if not pred_data:
        st.info("Insufficient longitudinal data for analytics. Initiate new diagnoses to populate charts.")
    else:
        # 1. Diagnostic Trends
        st.subheader("Diagnostic Volume Trends")
        df_preds = pd.DataFrame([p.model_dump() for p in pred_data])
        df_preds['Timestamp'] = pd.to_datetime(df_preds['timestamp'])
        df_preds['Outcome'] = df_preds['prediction']
        df_preds['Confidence'] = df_preds['confidence']
        df_preds['Date'] = df_preds['Timestamp'].dt.date

        trend_df = df_preds.groupby(['Date', 'Outcome']).size().unstack(fill_value=0)
        st.line_chart(trend_df)

        c1, c2 = st.columns(2)

        with c1:
            # 2. Confidence Distribution
            st.subheader("Confidence Distribution")
            # Clean confidence string (e.g. "95.5%") to float
            df_preds['ConfVal'] = df_preds['Confidence'].str.replace('%', '').astype(float)
            st.bar_chart(df_preds['ConfVal'].value_counts().sort_index())

        with c2:
            # 3. Outcome Ratio
            st.subheader("Outcome Distribution")
            outcome_counts = df_preds['Outcome'].value_counts()
            st.bar_chart(outcome_counts)

        # 4. Model Drift Telemetry
        st.divider()
        st.subheader("Model Drift Telemetry (System Health)")
        if drift_data:
            df_drift = pd.DataFrame([d.model_dump() for d in drift_data])
            df_drift['Timestamp'] = pd.to_datetime(df_drift['timestamp'])
            df_drift.set_index('Timestamp', inplace=True)
            st.line_chart(df_drift[['mean_val', 'std_val']])
            st.caption("Monitoring mean and standard deviation of input tensors to detect environmental shifts.")
        else:
            st.warning("No drift telemetry logs found. Automated health checks pending.")
