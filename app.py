import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Page title
# -----------------------------
st.set_page_config(page_title="Liver MRI Classifier", page_icon="🩺")

st.title("🩺 Liver Cancer MRI Classification")
st.write("Upload a liver MRI image and predict the disease class.")

# -----------------------------
# Class labels
# -----------------------------
class_names = [
    "Angiosarcoma",
    "Cholangiocarcinoma",
    "Healthy",
    "Hemangioma",
    "Hepatocellular Carcinoma"
]

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("liver_cancer_densenet121_model.keras")

model = load_model()

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess image
    img_array = preprocess_image(image)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction[0]) * 100

    st.subheader("Prediction Result")
    st.success(f"Predicted Class: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Class Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")

    # Bar chart
    fig, ax = plt.subplots()
    ax.bar(class_names, prediction[0] * 100)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Probability (%)")
    plt.title("Prediction Confidence")
    st.pyplot(fig)

st.markdown("---")
st.warning("⚠️ This application is for academic demonstration only.")
