import predictor
from PIL import Image
import streamlit as st

model = predictor.model
class_indices = predictor.class_indices

st.title("Plant Disease Classifer")

uploaded_image = st.file_uploader("Upload plant image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button("Classify Image"):
            predicton = predictor.predict_image_class(
                model, uploaded_image, class_indices
            )
            st.success(f"Prediction: {str(predicton)}")
