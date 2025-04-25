import streamlit as st
from PIL import Image
from helper import load_model, predict_fracture

st.title("🦴 Bone Fracture Detection from X-ray")
st.write("Upload an X-ray image and let the model predict if it shows a fracture.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner("Analyzing..."):
        model = load_model()
        prediction = predict_fracture(image, model)
        st.success(f"Prediction: {prediction}")
