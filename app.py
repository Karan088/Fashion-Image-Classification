import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from PIL import Image

# Load model & label names
model = load_model("fashion_mnist_cnn.h5")
with open("label_names.pkl", "rb") as f:
    label_names = pickle.load(f)

st.title("🧥 Fashion MNIST Classifier")
st.markdown("Upload a 28x28 grayscale image of a fashion item to predict its label.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L").resize((28, 28))
    img_array = np.array(img)
    st.image(img_array, caption="Uploaded Image", width=150)

    # Preprocess
    input_img = img_array.reshape(1, 28, 28) / 255.0
    prediction = model.predict(input_img)
    predicted_label = np.argmax(prediction)

    st.write("### Prediction:")
    st.success(f"{label_names[predicted_label]}")
