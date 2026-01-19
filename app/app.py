import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

from src.gradcam import GradCAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model
@st.cache_resource
def load_model(weights_path: str):
    model = models.efficientnet_b2(
        weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


# Preprocess
preprocess = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def quality_checks(pil_img: Image.Image):
    """Return (ok: bool, warnings: list[str]) using simple blur/brightness checks."""
    warnings = []
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Blur check
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 80:  # adjust if  strict/loose
        warnings.append("Image may be blurry (try retaking with better focus).")

    # Brightness check
    mean_brightness = gray.mean()
    if mean_brightness < 60:
        warnings.append("Image seems too dark (try brighter lighting).")
    if mean_brightness > 210:
        warnings.append("Image seems overexposed (reduce glare / avoid flash).")

    return (len(warnings) == 0), warnings

def overlay_heatmap(pil_img: Image.Image, cam_01: np.ndarray, alpha=0.45):
    """
    pil_img: original image
    cam_01: HxW cam in 0..1 (numpy)
    """
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    cam = cv2.resize(cam_01, (w, h))
    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
    return overlay

def risk_tier(p):
    # p = probability of class 1 
    if p < 0.30:
        return "Low"
    if p < 0.65:
        return "Medium"
    return "High"

# UI
st.set_page_config(page_title="DermaScan", layout="centered")
st.title("DermaScan — Skin Lesion Risk Triage")
st.caption("Educational demo only — **not a diagnosis**. If you’re concerned, consult a medical professional.")

weights_path = "models/best.pt"
if not os.path.exists(weights_path):
    st.error("Model weights not found at models/best.pt. Train first or put the file there.")
    st.stop()

model = load_model(weights_path)
gradcam = GradCAM(model)

uploaded = st.file_uploader("Upload a skin lesion photo (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")

    ok, warns = quality_checks(pil_img)
    if warns:
        st.warning("Photo quality notes:")
        for w in warns:
            st.write(f"- {w}")

    # Preprocess
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        p_susp = float(probs[1])

    tier = risk_tier(p_susp)

    st.subheader("Result")
    st.write(f"**Risk tier:** {tier}")
    st.write(f"**Suspicious probability:** {p_susp:.3f}")

    # Grad-CAM for class 1 
    cam_t = gradcam.generate(x, class_idx=1)  # torch [h,w] 0..1
    cam = cam_t.numpy()

    overlay = overlay_heatmap(pil_img, cam)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Original", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM heatmap overlay", use_container_width=True)

    st.markdown("---")
    st.subheader("Next steps (demo-safe)")
    if tier == "Low":
        st.write("- Consider monitoring for changes over time (size/color/border).")
    elif tier == "Medium":
        st.write("- Consider taking clearer photos and monitoring closely. If it changes, seek professional advice.")
    else:
        st.write("- Consider seeking professional evaluation, especially if the lesion is changing or concerning.")

    st.caption("This tool is just a demo. It does not provide medical diagnosis or treatment advice.")
