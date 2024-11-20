import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


# Load your trained model
MODEL_PATH ="D:\Spyder Work\moodvision_model.h5" # Replace with your actual model path
model = load_model(MODEL_PATH)



# Define a function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert('RGB')  # Convert to RGB (3 channels)
    img = img.resize((200, 200))  # Resize to match model input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    
    # Add batch dimension (shape: (1, 200, 200, 3))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Define the prediction function
def predict_emotion(img_array):
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "Happy 😊"
    else:
        return "Sad 😔"

# Streamlit app
st.title("Mood Vision 😊😔 Analysis")
st.write("Upload an image to detect if the person is Happy or Sad.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image",use_container_width=True)
    st.write("Processing...")

    # Preprocess and predict
    img_array = preprocess_image(uploaded_file)
    result = predict_emotion(img_array)

    # Display the result
    st.write(f"**Prediction:** {result}")
