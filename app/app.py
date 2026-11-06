
import sys
from pathlib import Path

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


# -------------------------------
# 1. Project paths and imports
# -------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add src folder to Python import path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import CNN_Autoencoder


# -------------------------------
# 2. Configuration
# -------------------------------

st.set_page_config(
    page_title="Hazelnut Anomaly Detection",
    page_icon="🌰",
    layout="wide"
)

# Image-level threshold selected using validation data
IMAGE_THRESHOLD = 0.019204156473279

# Pixel-level threshold selected using validation data
PIXEL_THRESHOLD = 0.013013083

DEVICE = torch.device("cpu")


# -------------------------------
# 3. Load trained model
# -------------------------------

@st.cache_resource
def load_model():

    model = CNN_Autoencoder()

    checkpoint_path = (
        PROJECT_ROOT
        / "models"
        / "cnn_autoencoder_experiment_1.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


model = load_model()


# -------------------------------
# 4. Image preprocessing
# -------------------------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# -------------------------------
# 5. Application interface
# -------------------------------

st.title("🌰 Hazelnut Anomaly Detection System")

st.write(
    "An AI-based industrial quality inspection prototype "
    "using a Convolutional Autoencoder."
)

st.info(
    "Upload a hazelnut image to identify potential defects "
    "and visualise anomalous regions."
)

st.markdown(
    """
    **Detection approach:** The system uses reconstruction-error-based
    anomaly detection. The image-level anomaly score is calculated
    using the average error of the highest-error 1% of pixels.
    """
)

uploaded_file = st.file_uploader(
    "Upload a hazelnut image",
    type=["jpg", "jpeg", "png"]
)


# -------------------------------
# 6. Prediction
# -------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        reconstruction = model(input_tensor)

    # Calculate pixel-level reconstruction error map
    error_map = torch.mean(
        (input_tensor - reconstruction) ** 2,
        dim=1
    ).squeeze().numpy()

    # -------------------------------
    # Top 1% anomaly scoring
    # -------------------------------

    errors = error_map.flatten()

    top1_count = max(
        1,
        int(errors.size * 0.01)
    )

    anomaly_score = float(
        np.sort(errors)[-top1_count:].mean()
    )

    # -------------------------------
    # Image-level classification
    # -------------------------------

    if anomaly_score >= IMAGE_THRESHOLD:

        prediction = "POTENTIAL DEFECT DETECTED"
        st.error(prediction)

    else:

        prediction = "NO ANOMALY DETECTED"
        st.success(prediction)

    # -------------------------------
    # Pixel-level anomaly detection
    # -------------------------------

    predicted_mask = error_map > PIXEL_THRESHOLD

    # Convert tensors to images
    original = (
        input_tensor.squeeze(0)
        .permute(1, 2, 0)
        .numpy()
    )

    reconstructed = (
        reconstruction.squeeze(0)
        .permute(1, 2, 0)
        .numpy()
    )

    # -------------------------------
    # 7. Display results
    # -------------------------------

    st.subheader("Inspection Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            original,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            reconstructed,
            caption="Reconstructed Image",
            use_container_width=True
        )

    with col3:
        fig, ax = plt.subplots()

        ax.imshow(error_map, cmap="jet")
        ax.axis("off")
        ax.set_title("Anomaly Heatmap")

        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------
    # 8. Anomaly score
    # -------------------------------

    st.subheader("Anomaly Score")

    st.metric(
        label="Top 1% Reconstruction Error",
        value=f"{anomaly_score:.6f}"
    )

    st.caption(
        f"Image classification threshold: {IMAGE_THRESHOLD:.8f}"
    )

    st.caption(
        f"Top 1% of pixels considered: {top1_count:,} / {errors.size:,}"
    )

    # -------------------------------
    # 9. Defect localisation
    # -------------------------------

    st.subheader("Defect Localisation")

    col4, col5 = st.columns(2)

    with col4:
        fig, ax = plt.subplots()

        ax.imshow(predicted_mask, cmap="gray")
        ax.set_title("Predicted Anomaly Regions")
        ax.axis("off")

        st.pyplot(fig)
        plt.close(fig)

    with col5:
        fig, ax = plt.subplots()

        ax.imshow(original)

        ax.imshow(
            error_map,
            cmap="jet",
            alpha=0.45
        )

        ax.set_title("Anomaly Overlay")
        ax.axis("off")

        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------
    # 10. Interpretation
    # -------------------------------

    st.subheader("Prediction Interpretation")

    if anomaly_score >= IMAGE_THRESHOLD:

        st.warning(
            "The Top 1% reconstruction error exceeds the "
            "validation-selected threshold. The image is flagged "
            "for further quality inspection."
        )

    else:

        st.success(
            "The Top 1% reconstruction error is below the "
            "validation-selected threshold. No anomaly was "
            "detected by the image-level classifier."
        )

    st.caption(
        "Note: This is an academic prototype. Predictions are "
        "not a substitute for human quality inspection. "
        "Performance may vary on images outside the MVTec "
        "Hazelnut dataset."
    )
