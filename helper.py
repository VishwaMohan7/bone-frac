from tensorflow.keras.models import load_model as keras_load
import numpy as np
from PIL import Image

def load_model():
    return keras_load("model/fracture_model.h5")

def predict_fracture(image, model):
    image = image.resize((224, 224))  # Resize to match model input
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    prediction = model.predict(img_array)[0][0]
    return "Fracture Detected" if prediction > 0.5 else "No Fracture"
