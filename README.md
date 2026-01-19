# DermaScan

DermaScan is a hackathon MVP for skin lesion risk triage using deep learning
(EfficientNet + Grad-CAM explainability).

This is an educational demo only — not a medical diagnostic tool.

## Features
- Upload skin lesion images
- Risk tier classification (Low / Medium / High)
- Grad-CAM visual explanations
- Image quality warnings

## Tech Stack
- PyTorch
- EfficientNet
- Grad-CAM
- Streamlit

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
