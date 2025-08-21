import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from label_map_kan import folder_num_to_kannada_map

# Load model
model = load_model("models/tulu_character_model_best1.h5")

# Prediction function
def predict_character(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)[0]

    # Folders are "1..49"
    folder_name = str(class_index + 1)

    # Map to Kannada
    kannada_char = folder_num_to_kannada_map.get(folder_name, "?")

    confidence = np.max(predictions) * 100
    return folder_name, kannada_char, confidence

# Streamlit UI
st.title("🖋️ Tulu to Kannada OCR")

uploaded_file = st.file_uploader("Upload a character image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with open("temp.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    folder_name, kannada_char, confidence = predict_character("temp.png")

    st.success(f"Predicted Folder: {folder_name}")
    st.info(f"Kannada Character: {kannada_char}")
    st.write(f"Confidence: {confidence:.2f}%")
