import streamlit as st
import numpy as np
from PIL import Image
import cv2
import hashlib

from model.inference import predict
from llm.telus_llm import generate_farmer_brief

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="TELUS FLOURISH",
    layout="wide"
)

# --------------------------------------------------
# UI STYLES
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #EEF6F0 0%, #F8FBF9 60%);
}

section[data-testid="stSidebar"] {
    background-color: #2F5D50;
    padding-top: 2rem;
}
section[data-testid="stSidebar"] * {
    color: white;
}

.card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
    margin-bottom: 1.5rem;
}

[data-testid="stFileUploader"] {
    background: white;
    border-radius: 16px;
    border: 2px dashed #CDE6D6;
    padding: 2rem;
}

img {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
logo = Image.open("assets/telus_flourish_logo.png")

col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image(logo, use_container_width=True)

with col_title:
    st.markdown("""
    <h1 style="color:#4B286D; margin-bottom:0;">TELUS FLOURISH</h1>
    <p style="font-size:18px; color:#5FB3A2;">
    Edge AI for Precision Agriculture & Sustainable Farming
    </p>
    """, unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "summary" not in st.session_state:
    st.session_state.summary = None
if "last_image_hash" not in st.session_state:
    st.session_state.last_image_hash = None

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
st.markdown("### 📸 Upload Drone or Field Image")

uploaded = st.file_uploader("", type=["jpg", "png", "jpeg"])

if not uploaded:
    st.info("Drag and drop an image here to begin analysis.")
    st.stop()

image = np.array(Image.open(uploaded).convert("RGB"))
image_hash = hashlib.md5(image.tobytes()).hexdigest()

# --------------------------------------------------
# MODEL INFERENCE
# --------------------------------------------------
with st.spinner("Running edge disease detection..."):
    prob_map, disease, confidence = predict(image)

heat = (prob_map * 255).astype("uint8")
heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
overlay = (image * 0.6 + heat * 0.4).astype("uint8")

high_conf = prob_map > 0.7
treated_pct = high_conf.mean() * 100
reduction = 100 - treated_pct

# --------------------------------------------------
# SPRAY ESTIMATION
# --------------------------------------------------
FIELD_SIZE_HA = 1.0
SPRAY_RATE_L_PER_HA = 200

spray_area_ha = FIELD_SIZE_HA * (treated_pct / 100)
spray_volume_l = spray_area_ha * SPRAY_RATE_L_PER_HA

# --------------------------------------------------
# DISEASE → PESTICIDE MAP
# --------------------------------------------------
DISEASE_MAP = {
    "Leaf Disease": {
        "disease": "Fungal Leaf Blight",
        "pesticide": "Azoxystrobin-based fungicide"
    }
}

info = DISEASE_MAP.get(disease, {
    "disease": disease,
    "pesticide": "Crop-specific fungicide"
})

# --------------------------------------------------
# AUTO SUMMARY (ON IMAGE CHANGE)
# --------------------------------------------------
if st.session_state.last_image_hash != image_hash:
    st.session_state.summary = generate_farmer_brief({
        "disease": info["disease"],
        "treated_area_pct": treated_pct,
        "chemical_reduction_pct": reduction,
        "confidence": confidence,
        "spray_area_ha": spray_area_ha,
        "spray_volume_l": spray_volume_l,
        "pesticide": info["pesticide"]
    })
    st.session_state.last_image_hash = image_hash

# --------------------------------------------------
# SIDEBAR — METRICS ONLY
# --------------------------------------------------
st.sidebar.header("🌱 Impact Metrics")
st.sidebar.metric("Affected Area (%)", f"{treated_pct:.2f}")
st.sidebar.metric("Chemical Reduction (%)", f"{reduction:.2f}")
st.sidebar.metric("Spray Volume (L)", f"{spray_volume_l:.1f}")
st.sidebar.metric("Model Confidence", f"{confidence:.2f}")

# --------------------------------------------------
# VISUAL RESULTS
# --------------------------------------------------
st.markdown("### 🧠 Disease Detection Results")

c1, c2 = st.columns(2)
c1.image(image, caption="Original Field Image", use_container_width=True)
c2.image(overlay, caption="Disease Probability Heatmap", use_container_width=True)

# --------------------------------------------------
# FARMER SUMMARY
# --------------------------------------------------
st.markdown("### 🧾 Farmer Summary")
st.markdown(f"<div class='card'>{st.session_state.summary}</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FARMER DECISION + POPUP
# --------------------------------------------------
st.markdown("### 🚜 Farmer Decision")

decision = st.radio(
    "Do you approve precision spraying for this field?",
    ["Approve Precision Spraying", "Delay / Reject"],
    index=None
)

if decision:
    if decision.startswith("Approve"):
        st.toast("✅ Precision spraying approved", icon="🚜")
    else:
        st.toast("⏸️ Spraying delayed", icon="🕒")

st.caption("TELUS FLOURISH | Edge AI + TELUS LLMs | Hackathon Demo")
