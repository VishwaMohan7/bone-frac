import tensorflow as tf
import numpy as np
from PIL import Image

def load_model():
    # Ensure you're loading the TensorFlow Lite model (.tflite)
    model_path = 'model/fracture_model.tflite'  # Ensure correct path to your .tflite model

    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    return interpreter

def predict_fracture(image, interpreter):
    # Preprocess the image to match the model's input size (e.g., 224x224)
    image = image.resize((224, 224))  # Resize to match model input size
    image_array = np.array(image, dtype=np.float32)
    
    # Normalize the image data if necessary (e.g., scale to 0-1 range)
    image_array = image_array / 255.0
    
    # Add batch dimension (model expects a batch of images)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Set input tensor for the TFLite model
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], image_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Assuming the model's output is a single value indicating fracture prediction
    prediction = output_data[0][0]  # Adjust this according to your model output shape
    
    return prediction
