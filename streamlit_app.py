import streamlit as st
import subprocess
import os
from PIL import Image
import numpy as np
from tensorflow.keras.layers import SeparableConv2D
import tensorflow as tf

# Install dependencies
# subprocess.call(["pip", "install", "-r", "./requirements.txt"])

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors

st.set_page_config(page_title="Fruit Detection", layout="wide")

# Define labels (adjust according to your model's classes)
labels = ['fresh', 'rotten']

# Custom SeparableConv2D to ignore unsupported arguments
class CustomSeparableConv2D(SeparableConv2D):
    def __init__(self, *args, **kwargs):
        # Remove unsupported arguments if present
        kwargs.pop('groups', None)
        kwargs.pop('kernel_initializer', None)
        kwargs.pop('kernel_regularizer', None)
        kwargs.pop('kernel_constraint', None)
        super().__init__(*args, **kwargs)

# Load model
@st.cache_resource  # Cache the model loading
def load_model():
    custom_objects = {'SeparableConv2D': CustomSeparableConv2D}
    model = tf.keras.models.load_model('./pretrained_fruit_classification.keras', custom_objects=custom_objects, compile=False)
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()
st.success("Model loaded successfully!")

# Preprocess the image
def preprocess_image(image):
    """Preprocess the image to the required input shape for the model."""
    image = image.resize((224, 224))  # Adjust size according to model's input size
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Predict the label
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    return prediction

# Streamlit app
st.title("Fruit Detection 🍎🍌🍊")
st.write("Upload multiple images and view them in a grid layout.")

uploaded_files = st.file_uploader(
    "Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)


if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} image(s) successfully!")

    # Number of columns in the grid
    num_columns = 6
    
    # Calculate the number of rows needed
    num_rows = len(uploaded_files) // num_columns + int(len(uploaded_files) % num_columns > 0)
    
    # Loop through rows
    for row in range(num_rows):
        cols = st.columns(num_columns)
        
        # Loop through each column in the current row
        for col in range(num_columns):
            index = row * num_columns + col
            if index < len(uploaded_files):
                with cols[col]:
                    uploaded_file = uploaded_files[index]
                    try:
                        # Open and display the image
                        image = Image.open(uploaded_file)

                        # Perform prediction
                        prediction_prob = predict(image)

                        predicted_index = np.round(prediction_prob).astype(int)
                        predicted_label = labels[predicted_index]

                        predicted_label_color = "#00884A" if predicted_label == 'fresh' else "#FF5152"
                        
                        st.image(image, use_column_width=True)

                        box_html = f"""
                        <div style="
                            margin: 0px; 
                            font-weight: 500;
                            margin-bottom: 40px;
                        ">
                            <table style='width: 100%; border-spacing: 0; margin: 0;'>
                                <tr>
                                    <td style='font-weight: bold; font-size: 14px; color: #555;'>Prediction:</td>
                                    <td style='font-weight: bold; font-size: 16px; color: {predicted_label_color}; text-align: right;'>{predicted_label.upper()}</td>
                                </tr>
                                <tr>
                                    <td style='font-weight: bold; font-size: 14px; color: #555;'>Confidence Score:</td>
                                    <td style='font-weight: bold; font-size: 16px; color: #007BC0; text-align: right;'>{prediction_prob*100:.2f}%</td>
                                </tr>
                            </table>
                        </div>
                    """

                        st.markdown(box_html, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"An error occurred with Image {index + 1}: {e}")
else:
    st.info("Upload images to display them in a grid with predictions.")