import os

import pandas as pd
import streamlit as st

from pneumonia_classifier.utils.database import get_patient_history


def render_history_tab():
    st.header("Diagnostic Audit History")
    search_id = st.text_input("Search Patient History", value="P-DEFAULT", key="history_search_id")
    if st.button("Fetch Logs", key="fetch_history_btn"):

        history = get_patient_history(search_id)
        if history:
            # Map Pydantic models to a list of dicts for the DataFrame
            data = [h.model_dump() for h in history]
            df = pd.DataFrame(data)

            # Map internal field names to user-friendly column names
            df = df.rename(columns={
                "timestamp": "Timestamp",
                "patient_id": "Patient",
                "prediction": "Outcome",
                "confidence": "Conf",
                "heatmap_path": "Path",
                "requester_id": "Doc",
                "requester_ip": "IP"
            })

            # Make Path clickable - maintaining the existing query param logic
            def make_clickable(path):
                if path and os.path.exists(path):
                    # We use query params for viewer-mode (captured in streamlit_app.py)
                    return f'<a href="?viewer={path}" target="_self">View Heatmap</a>'
                return "N/A"

            df['Path'] = df['Path'].apply(make_clickable)

            # Render as HTML to support links
            st.write(df[["Timestamp", "Patient", "Outcome", "Conf", "Path", "Doc", "IP"]].to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("No records found for this identifier.")
