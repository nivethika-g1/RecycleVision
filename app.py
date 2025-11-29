import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

st.set_page_config(page_title="RecycleVision", page_icon="♻")
st.title("♻ RecycleVision – Garbage Image Classification")
st.write(
    "Upload an image of waste and the model will classify it into one of the six categories: "
    "cardboard, glass, metal, paper, plastic or trash."
)

# Load trained model
@st.cache_resource
def load_garbage_model():
    model = load_model("garbage_classifier.h5")
    return model

model = load_garbage_model()

# Class labels in the SAME order as train_data.class_indices
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds)
    predicted_label = labels[predicted_index]
    confidence = float(np.max(preds)) * 100

    st.subheader(f"Predicted Category: **{predicted_label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Optional: show all class probabilities
    st.markdown("### Class probabilities:")
    for i, label in enumerate(labels):
        st.write(f"- {label}: {preds[0][i]*100:.2f}%")
else:
    st.info("Please upload a JPG/PNG image to get a prediction.")
