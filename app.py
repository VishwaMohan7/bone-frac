import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from helper import load_model, predict_fracture

# Streamlit interface
st.title("Bone Fracture Detection")

# Load the TensorFlow Lite model with error handling
try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()  # This will stop the app if model fails to load

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Open the image
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Predict the fracture status
        with st.spinner("Analyzing..."):
            prediction = predict_fracture(image, model)
            
            # Display prediction with confidence
            confidence = prediction[0][0] if isinstance(prediction, np.ndarray) else prediction
            st.metric("Prediction Result", 
                     "Fracture detected" if confidence > 0.5 else "No fracture detected",
                     f"{confidence:.2%} confidence")
            
            # Optional: show raw prediction value
            with st.expander("See raw prediction"):
                st.write(f"Raw prediction value: {confidence:.4f}")
                
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
