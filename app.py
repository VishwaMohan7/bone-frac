import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from helper import load_model, predict_fracture

# Set page configuration
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="💀",
    layout="wide",
)

# Header Section with Styling
st.markdown(
    """
    <style>
    .header {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
    }
    .subheader {
        font-size: 18px;
        text-align: center;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="header">Bone Fracture Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload an X-ray image, and let the AI detect fractures with precision.</div>', unsafe_allow_html=True)

# Sidebar for Model Loading
st.sidebar.title("⚙️ Settings")
try:
    with st.spinner("Loading Model..."):  # Corrected spinner usage
        model = load_model()
    st.sidebar.success("Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"Model Loading Failed: {str(e)}")
    st.stop()

# Upload Section
st.markdown("### 📤 Upload an X-ray Image")
uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Display image in a column layout
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("#### Uploaded Image:")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

        # Predict the fracture status
        with col2:
            st.markdown("#### Prediction Result:")
            with st.spinner("Analyzing the X-ray image..."):
                prediction = predict_fracture(image, model)
                confidence = prediction[0][0] if isinstance(prediction, np.ndarray) else prediction
                result = "No Fracture Detected" if confidence > 0.5 else "Fracture Detected"
                st.metric(label="Result", value=result, delta=f"{confidence:.2%} confidence")

        # Raw prediction details
        with st.expander("📊 See Raw Prediction Details"):
            st.write(f"Raw prediction value: {confidence:.4f}")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.info("Please upload an image to get started.")

# Footer Section
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 14px; color: #aaa;">
        Made with ❤️ by [VishwaMohan7](https://github.com/VishwaMohan7)
    </div>
    """,
    unsafe_allow_html=True,
)
