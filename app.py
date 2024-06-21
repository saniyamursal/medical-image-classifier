import streamlit as st
from keras.models import load_model as keras_load_model
from PIL import Image, ImageOps
import numpy as np

# Function to load the model
@st.cache_resource
def load_keras_model():
    model = keras_load_model("keras_model.h5", compile=False)
    return model

# Load the labels
@st.cache_data
def load_labels():
    class_names = open("labels.txt", "r").readlines()
    return class_names

model = load_keras_model()
class_names = load_labels()

st.write("""
         # Image Classification
         """
         )

file = st.file_uploader("Please upload an image", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    index = np.argmax(predictions)
    class_name = class_names[index]
    confidence_score = predictions[0][index]

    st.success(f"Class: {class_name[2:].strip()}")
    st.success(f"Confidence Score: {confidence_score:.2f}")