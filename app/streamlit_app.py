"""
DefectLens AI — Interactive Streamlit Inspection Dashboard.

Provides an interactive web interface for industrial defect inspection.
Upload an image to get a defect classification, confidence score,
and anomaly heatmap visualization.

Launch:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import streamlit as st
import torch
from PIL import Image

# --- Page Config ---
st.set_page_config(
    page_title="DefectLens AI — Industrial Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .defective-badge {
        background: #ff4d4d;
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .normal-badge {
        background: #28a745;
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    checkpoint_path = st.text_input(
        "Model Checkpoint",
        value="",
        placeholder="outputs/bottle/best_model.pth",
        help="Path to a trained model checkpoint (.pth file)",
    )

    image_size = st.slider("Input Image Size", min_value=128, max_value=512, value=256, step=32)
    threshold = st.slider("Decision Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **DefectLens AI** detects manufacturing defects in industrial product images using deep learning.

    **Supported defect types:**
    - Scratches
    - Cracks
    - Dents
    - Surface irregularities
    - Contamination

    **Dataset:** [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
    """)

# --- Main Header ---
st.markdown('<div class="main-header">🔍 DefectLens AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Industrial Visual Inspection & Anomaly Detection</div>', unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model(checkpoint: str, img_size: int, thresh: float):
    """Load the inference model (cached)."""
    if not checkpoint:
        return None, "no_model"
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        return None, "not_found"
    try:
        from defectlens.inference.predictor import DefectPredictor
        device = torch.device("cpu")
        predictor = DefectPredictor(
            checkpoint_path=ckpt_path,
            image_size=img_size,
            threshold=thresh,
            device=device,
        )
        return predictor, "ok"
    except Exception as e:
        return None, f"error: {e}"


model, model_status = load_model(checkpoint_path, image_size, threshold)

# Model status indicator
col_stat1, col_stat2, col_stat3 = st.columns([1, 1, 3])
with col_stat1:
    if model_status == "ok":
        st.success("✅ Model loaded")
    elif model_status == "no_model":
        st.info("ℹ️ No model — Demo mode")
    elif model_status == "not_found":
        st.warning("⚠️ Checkpoint not found")
    else:
        st.error(f"❌ {model_status}")

st.markdown("---")

# --- Image Upload ---
st.markdown("### 📁 Upload Image for Inspection")
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    label_visibility="collapsed",
)

# --- Inference & Display ---
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_image)

    col_img, col_results = st.columns([1.2, 1])

    with col_img:
        st.markdown("#### Original Image")
        st.image(pil_image, use_container_width=True)
        st.caption(f"Size: {pil_image.size[0]}×{pil_image.size[1]} px | Format: {pil_image.format or uploaded_file.type}")

    with col_results:
        st.markdown("#### Inspection Result")

        if model is not None and model_status == "ok":
            with st.spinner("Running inspection..."):
                try:
                    result = model.predict(img_array)
                    label_str = "DEFECTIVE" if result.label == 1 else "NORMAL"
                    badge_class = "defective-badge" if result.label == 1 else "normal-badge"

                    st.markdown(
                        f'<div style="text-align:center; margin:1rem 0;">'
                        f'<span class="{badge_class}">{label_str}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Confidence", f"{result.confidence:.1%}")
                    m2.metric("Latency", f"{result.latency_ms:.1f} ms")
                    m3.metric("FPS", f"{1000/result.latency_ms:.0f}")

                    st.progress(result.confidence, text=f"Model confidence: {result.confidence:.1%}")

                except Exception as e:
                    st.error(f"Inference failed: {e}")
        else:
            # Demo mode — show placeholder
            st.markdown('<div class="info-box">ℹ️ <b>Demo Mode</b><br>Load a trained model checkpoint in the sidebar to run real inference.</div>', unsafe_allow_html=True)

            import random
            demo_label = random.choice(["NORMAL", "DEFECTIVE"])
            demo_conf = random.uniform(0.7, 0.99)
            badge_class = "defective-badge" if demo_label == "DEFECTIVE" else "normal-badge"
            st.markdown(
                f'<div style="text-align:center; margin:1rem 0;">'
                f'<span class="{badge_class}">{demo_label} (demo)</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{demo_conf:.1%}", help="Demo value")
            m2.metric("Latency", "~15 ms", help="Demo value")
            m3.metric("FPS", "~67", help="Demo value")
            st.progress(demo_conf)

        # Heatmap placeholder (would show actual anomaly map from model)
        st.markdown("#### Anomaly Heatmap")
        if model is not None and model_status == "ok":
            st.info("Anomaly heatmap visualization requires anomaly detection model.")
        else:
            # Synthetic heatmap for demo
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            synthetic_heat = np.random.rand(32, 32)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(synthetic_heat, cmap="jet", vmin=0, vmax=1)
            ax.set_title("Anomaly Score Map (demo)", fontsize=10)
            ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption("🔴 High anomaly score  ·  🔵 Low anomaly score")

else:
    # Empty state
    st.markdown("""
    <div class="info-box">
    📌 <b>Getting Started</b><br>
    Upload a product image above to run defect inspection.<br><br>
    <b>Supported formats:</b> PNG, JPEG, BMP, TIFF<br>
    <b>Best results:</b> Well-lit images with consistent background
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("### Capabilities")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **🔍 Defect Detection**
        Classify products as normal or defective with confidence scores
        """)
    with c2:
        st.markdown("""
        **🗺️ Anomaly Heatmap**
        Visualize regions with highest anomaly likelihood
        """)
    with c3:
        st.markdown("""
        **📊 Performance Metrics**
        Real-time latency, FPS, and confidence reporting
        """)

# --- Footer ---
st.markdown("---")
st.markdown(
    "**DefectLens AI** · Built by [Habimana Moise](https://github.com/moise744) · "
    "[GitHub](https://github.com/moise744/defectlens-ai)",
    unsafe_allow_html=False,
)
