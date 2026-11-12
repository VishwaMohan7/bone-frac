# Bone Fracture Detection System
### X-ray Image Classification with TensorFlow Lite

A Streamlit application that lets users upload an X-ray image and run inference with a TensorFlow Lite model to produce a fracture / no-fracture classification.

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-306998?style=flat)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

</div>

---

## Overview

This project provides a simple web interface for running bone-fracture image classification on uploaded X-ray images. The application loads a TensorFlow Lite model, preprocesses the uploaded image, invokes the model, and displays the resulting classification in the Streamlit interface.

The current implementation uses a binary threshold of **0.5**:
- Output greater than 0.5 → **No Fracture Detected**
- Output less than or equal to 0.5 → **Fracture Detected**

The displayed output is the model's raw prediction-derived value; it should not be interpreted as a clinically validated probability.

## Features

- Upload X-ray images in PNG, JPG, or JPEG format.
- Preview the uploaded image in the dashboard.
- Resize the image to 224 × 224 pixels before inference.
- Normalize image pixel values to the 0–1 range.
- Run inference using a TensorFlow Lite interpreter.
- Display the predicted class and raw prediction value.
- Inspect the raw output in an expandable details section.

## Application Workflow

1. Upload an X-ray image through the Streamlit interface.
2. Preview the selected image.
3. Resize the image to 224 × 224 pixels.
4. Convert the image to a NumPy array and normalize pixel values.
5. Add a batch dimension and pass the tensor to the TensorFlow Lite interpreter.
6. Invoke the model and retrieve its output tensor.
7. Apply the 0.5 threshold to display the classification.

## Architecture

```text
       X-ray Image
            |
            v
     Streamlit Interface
            |
            v
      Image Preprocessing
    (224x224, float32, /255)
            |
            v
    TensorFlow Lite Model
            |
            v
       Model Output
            |
            v
     Threshold at 0.5
            |
            v
 Fracture / No Fracture Result
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Interactive web interface |
| TensorFlow Lite | Model loading and inference |
| NumPy | Image array preparation |
| Pillow (PIL) | Image loading and resizing |

## Repository Structure

```text
bone-frac/
├── app.py                         # Streamlit application
├── helper.py                      # Model loading and prediction helpers
├── requirements.txt               # Python dependencies
└── model/
    └── bone_fracture_model.tflite # TensorFlow Lite model
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/VishwaMohan7/bone-frac.git
cd bone-frac
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal. Open that URL in your browser, upload an X-ray image, and view the model output.

## Model Details

- **Format:** TensorFlow Lite (`.tflite`)
- **Model file:** `model/bone_fracture_model.tflite`
- **Input preprocessing in the application:** resize to 224 × 224, convert to `float32`, and divide pixel values by 255.
- **Decision threshold in the application:** 0.5.

The model's training dataset, evaluation metrics, architecture, and clinical validation details are not documented in this repository. They should be added only when verified.

## Important Medical Disclaimer

This project is an educational / experimental machine-learning application. It is **not a medical device**, does not provide a diagnosis, and must not be used to make treatment decisions. X-ray interpretation should be performed by qualified healthcare professionals. A model prediction may be incorrect even when the interface displays a confident-looking result.

---

<div align="center">

### Exploring the application of computer vision to medical imaging.

</div>
