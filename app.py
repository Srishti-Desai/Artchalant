import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="AI vs Human Art Detector")

# Load model (CHANGE NAME if needed)
model = load_model("model.h5")

st.title("AI vs Human Art Detector")

uploaded_file = st.file_uploader("Upload an artwork", type=["jpg","jpeg","png"])

def preprocess(image):
    image = image.resize((224,224))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    processed = preprocess(image)
    prediction = model.predict(processed)[0][0]

    if prediction > 0.5:
        st.success("Ai Art")
        st.write(f"Confidence: {prediction:.2%}")
    else:
        st.error("Human art")
        st.write(f"Confidence: {(1-prediction):.2%}")
