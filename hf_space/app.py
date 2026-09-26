
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent

with open(ROOT / "model_meta.json", "r") as f:
    META = json.load(f)

CLASS_NAMES = META["class_names"]
IMG_SIZE = int(META["img_size"])
MODEL_TYPE = META["model_type"]
ARCH = META["arch"]

st.set_page_config(
    page_title="Solar Panel Condition Classifier",
    page_icon="☀️",
    layout="centered",
)

st.title("☀️ Solar Panel Condition Classifier")
st.write(
    "Upload a visible-light RGB image of a photovoltaic panel. "
    "The model predicts one of six panel conditions and reports a confidence score."
)

@st.cache_resource
def load_model():
    if MODEL_TYPE == "ultralytics":
        from ultralytics import YOLO
        return YOLO(str(ROOT / "model.pt"))

    checkpoint = torch.load(ROOT / "model.pt", map_location="cpu")

    if ARCH == "ResNet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(in_features, len(CLASS_NAMES)),
        )

    elif ARCH == "EfficientNetB0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(in_features, len(CLASS_NAMES)),
        )
    else:
        raise ValueError(f"Unsupported architecture: {ARCH}")

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

MODEL = load_model()

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

def predict(image):
    image = image.convert("RGB")

    if MODEL_TYPE == "ultralytics":
        result = MODEL.predict(source=image, imgsz=IMG_SIZE, verbose=False)[0]
        raw = result.probs.data.detach().cpu().numpy()
        aligned = np.zeros(len(CLASS_NAMES), dtype=np.float32)

        for raw_idx, class_name in result.names.items():
            aligned[CLASS_NAMES.index(class_name)] = raw[int(raw_idx)]

        probs = aligned

    else:
        x = EVAL_TRANSFORM(image).unsqueeze(0)
        with torch.no_grad():
            logits = MODEL(x)
            probs = torch.softmax(logits, dim=1)[0].numpy()

    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs

uploaded = st.file_uploader(
    "Upload a solar-panel image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    label, confidence, probs = predict(image)

    st.subheader(f"Prediction: {label}")
    st.metric("Confidence", f"{confidence * 100:.2f}%")

    prob_df = pd.DataFrame({
        "Condition": CLASS_NAMES,
        "Probability": probs,
    }).sort_values("Probability", ascending=False)

    st.dataframe(
        prob_df.assign(Probability=lambda d: (d["Probability"] * 100).round(2)),
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(prob_df.set_index("Condition"))

st.divider()
st.caption(
    "Academic prototype only. It classifies visible RGB images and is not a certified "
    "electrical-safety, maintenance or industrial inspection system. Predictions should "
    "not replace qualified inspection."
)
st.caption(
    f"Model: {META['selected_model']} | "
    f"Training dataset: {META['dataset_name']} by {META['dataset_creator']} | "
    f"Dataset licence recorded for the research: {META['dataset_licence']}"
)
