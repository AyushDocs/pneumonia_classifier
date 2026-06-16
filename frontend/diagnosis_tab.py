import base64
import io
import logging
import os
import time

import streamlit as st
from PIL import Image

from frontend.utils import auto_crop_xray, is_valid_xray
from pneumonia_classifier.config import TRANSFORM, config
from pneumonia_classifier.utils.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


def render_diagnosis_tab():
    st.header("Pneumonia Diagnostic Suite")

    with st.expander("How this AI works"):
        st.write("""
        This system utilizes a **Custom CNN (Net)** architecture optimized for medical imaging:
        - **Texture Analysis**: Specialized in local texture patterns associated with lung consolidation.
        - **Feature Extraction**: Efficiently identifies diagnostic markers across the lung fields.
        - **Explainability**: Integrated with Grad-CAM to provide clinical focus maps for infectious findings.

        The current model provides high-speed, reliable diagnostic support focused on chest radiograph classification.
        """)

    col1, _ = st.columns([2, 1])
    with col1:
        st.markdown("### 1. Patient & Image Input")
        patient_id = st.text_input("Patient Identifier", value="P-DEFAULT", key="diagnosis_patient_id")

        # --- SAMPLE SELECTION ---
        st.markdown("#### Selection Method")
        input_method = st.radio("Choose Input Method", ["Upload Image", "Use Sample X-ray"], horizontal=True)

        image_to_process = None
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg", "dcm"])
            if uploaded_file:
                image_to_process = Image.open(uploaded_file).convert('RGB')
        else:
            sample_options = {
                "Normal Case": "data/samples/normal_sample.png",
                "Pneumonia Case": "data/samples/pneumonia_sample.png"
            }
            sample_choice = st.selectbox("Select Sample", list(sample_options.keys()))
            if os.path.exists(sample_options[sample_choice]):
                image_to_process = Image.open(sample_options[sample_choice]).convert('RGB')
            else:
                st.warning("Sample images not found. Please upload manually.")

        if image_to_process:
            st.image(image_to_process, caption="Patient Radiograph", width='stretch')

            if st.button("Initiate AI Analysis", key="analyze_btn"):
                render_inference_flow(image_to_process, patient_id)

        # --- DISPLAY RESULTS & REPORT ---
        if st.session_state.prediction_result:
            render_results(st.session_state.prediction_result)

def _run_local_inference(processed_image, patient_id):
    """Fallback: run inference directly in the Streamlit process when Celery is unavailable."""
    import cv2
    import numpy as np
    import torch
    from pneumonia_classifier.ml.explainability import get_medical_heatmap
    from pneumonia_classifier.ml.model.arch import Net
    from pneumonia_classifier.utils.database import save_drift_log, save_prediction

    @st.cache_resource
    def _load_model():
        model = Net()
        from pneumonia_classifier.config import get_model_path
        loaded_data = torch.load(get_model_path(), map_location='cpu', weights_only=False)
        if not isinstance(loaded_data, dict) and hasattr(loaded_data, "state_dict"):
            state_dict = loaded_data.state_dict()
        else:
            state_dict = loaded_data
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    model = _load_model()
    input_tensor = TRANSFORM(processed_image).unsqueeze(0)

    # Drift logging
    try:
        save_drift_log(f"local_{int(time.time())}", float(np.mean(input_tensor.numpy())), float(np.std(input_tensor.numpy())))
    except Exception:
        pass

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.exp(output)

    pred_idx = int(probabilities.argmax().item())
    prediction = "Pneumonia" if pred_idx == 1 else "Normal"
    confidence = f"{probabilities[0][pred_idx].item() * 100:.1f}%"

    # Grad-CAM
    heatmap_base64 = ""
    heatmap_path = ""
    try:
        heatmap_img_bgr = get_medical_heatmap(model, input_tensor, processed_image)
        if heatmap_img_bgr is not None:
            heatmap_path = f"data/heatmaps/{patient_id}_{int(time.time())}.png"
            os.makedirs("data/heatmaps", exist_ok=True)
            cv2.imwrite(heatmap_path, heatmap_img_bgr)
            h_rgb = cv2.cvtColor(heatmap_img_bgr, cv2.COLOR_BGR2RGB)
            buf = io.BytesIO()
            Image.fromarray(h_rgb).save(buf, format="PNG")
            heatmap_base64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    except Exception as e:
        logger.warning(f"Grad-CAM failed: {e}")

    # Original image as base64
    orig_buf = io.BytesIO()
    processed_image.save(orig_buf, format="PNG")
    orig_b64 = f"data:image/png;base64,{base64.b64encode(orig_buf.getvalue()).decode('utf-8')}"

    save_prediction(patient_id, prediction, confidence, heatmap_path, st.session_state.user, "127.0.0.1")

    return {
        "patient_id": patient_id,
        "prediction": prediction,
        "confidence": confidence,
        "heatmap_base64": heatmap_base64,
        "original_image": orig_b64,
    }


def render_inference_flow(image_to_process, patient_id):
    with st.spinner("Executing CNN Inference..."):
        valid, reason = is_valid_xray(image_to_process)
        if not valid:
            st.warning(f"OOD ALERT: {reason}")
        else:
            processed_image = auto_crop_xray(image_to_process)
            REDIS_URL = config.REDIS_URL

            # Quick socket check — avoids Celery's background retry threads
            import socket
            redis_available = False
            try:
                from urllib.parse import urlparse
                parsed = urlparse(REDIS_URL)
                sock = socket.create_connection((parsed.hostname, parsed.port), timeout=2)
                sock.close()
                redis_available = True
            except Exception:
                redis_available = False

            if redis_available:
                try:
                    from celery import Celery
                    celery_app = Celery("inference_tasks", broker=REDIS_URL, backend=REDIS_URL)
                    celery_app.conf.broker_connection_retry_on_startup = False

                    buffered = io.BytesIO()
                    processed_image.save(buffered, format="PNG")
                    b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    requester_ip = "127.0.0.1"
                    job_id = f"job_{int(time.time())}"

                    celery_app.send_task(
                        "process_inference",
                        args=[job_id, b64_image, patient_id, st.session_state.user, requester_ip]
                    )

                    report_url = f"/?job={job_id}"
                    st.info(f"**Analysis in Progress.** [View Live Diagnostic Report]({report_url})")
                    return

                except Exception:
                    pass

            # Fallback: local inference
            st.warning("**Celery/Redis unavailable** — running inference locally. "
                       "Start Redis + Celery worker for better throughput and background processing.")
            result = _run_local_inference(processed_image, patient_id)
            st.session_state.prediction_result = result
            st.rerun()

def render_results(res):
    st.markdown("---")
    st.subheader("Diagnostic Assessment")
    m1, m2 = st.columns(2)
    m1.metric("Prediction", res["prediction"])
    m2.metric("Confidence Level", res["confidence"])

    if res["prediction"] == "Pneumonia":
        st.error("Radiographical findings suggestive of infectious consolidation.")
    else:
        st.success("Lung fields appear clear of major infectious markers.")

    with st.expander("Confidence & Metrics Context"):
        st.info("""
        **Confidence Interval Explanation**: While the overall model achieves >90% precision and recall on the test suite,
        this specific result represents the AI's confidence for this individual scan. Normally, clinical results fall within
        high-confidence ranges, but edge cases or artifacts may yield lower localized intervals.
        """)

    st.markdown("### 1. Visual Evidence")
    v1, v2 = st.columns(2)

    with v1:
        st.markdown("**Original X-ray Scan**")
        orig_img = res.get("original_image")
        if orig_img:
            st.image(orig_img, use_container_width=True)
        else:
            st.warning("Original scan data not available in results.")

    with v2:
        if res["prediction"] == "Pneumonia":
            st.markdown("**AI Focus Analysis (Grad-CAM)**")
            h_b64 = res.get("heatmap_base64")
            if h_b64:
                st.image(h_b64, use_container_width=True)
            else:
                st.info("Neural focus map pending generation.")

    if res["prediction"] == "Pneumonia":
        with st.expander("Understanding the AI focus (Grad-CAM)"):
            st.write("""
            **Grad-CAM** (Gradient-weighted Class Activation Mapping) highlights the regions the AI prioritized.
            In this case, it focused on areas of radiographical density supporting a pneumonia diagnosis.
            """)

    st.markdown("### 2. Clinical Documentation")
    if not os.path.exists(config.REPORT_TEMP_DIR):
        os.makedirs(config.REPORT_TEMP_DIR, exist_ok=True)

    report_filename = f"report_{res['patient_id']}.pdf"
    report_path = os.path.join(config.REPORT_TEMP_DIR, report_filename)
    try:
        ReportGenerator.generate_clinical_report(
            res['patient_id'], res['prediction'], res['confidence'],
            res.get("heatmap_base64", ""), res.get("original_image", ""), report_path
        )

        with open(report_path, "rb") as f:
            st.download_button(
                label="Download Clinical PDF Report",
                data=f,
                file_name=report_filename,
                mime="application/pdf",
                key="download_report_btn"
            )
    except Exception as e:
        st.error(f"Report Engine Error: {str(e)}")
