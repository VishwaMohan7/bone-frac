import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from helper import load_model, predict_fracture

# Load the TensorFlow Lite model
model = load_model()

# Streamlit interface
st.title("Bone Fracture Detection")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict the fracture status
    with st.spinner("Analyzing..."):
        prediction = predict_fracture(image, model)
        
        # Check prediction output and display the result
        if prediction > 0.5:  # Assuming 0.5 is the threshold for fracture
            st.success("Prediction: Fracture detected")
        else:
            st.success("Prediction: No fracture detected")
