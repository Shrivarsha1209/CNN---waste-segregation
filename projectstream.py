import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the model
model = tf.keras.models.load_model("mymodel.h5")

# Define the predict function
def predict(image):
    # Make predictions using the loaded model
    prediction = model.predict(image)
    return prediction

# Define preprocess_image function
def preprocess_image(image):
    # Open the image using PIL
    img = Image.open(image)
    # Resize the image to match the input size of the model (150x150)
    img = img.resize((150, 150))
    # Convert the image to numpy array
    img_array = np.array(img)
    # Preprocess the image for MobileNetV2 model
    img_array = preprocess_input(img_array)
    # Expand the dimensions of the image to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define class labels
# Define class labels
# Define class labels
class_labels = {
    0: "Organic Waste",
    1: "Recyclable Waste",
    2: "E-Waste"
}

# Create the Streamlit app
def main():
    st.title('Waste classification app')
    uploaded_image = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_image is not None:
        image = preprocess_image(uploaded_image)
        prediction = predict(image)
        # Get the index of the predicted class
        predicted_class_idx = np.argmax(prediction)
        # Map the predicted class index to label
        predicted_label = class_labels[predicted_class_idx]
        st.write("Predicted Class:", predicted_label)

if __name__ == "__main__":
    main()

