import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# --- App Configuration ---
# Set the page configuration for a better layout
st.set_page_config(page_title="Tulu Script Translator", layout="wide")

# --- Model and Data Configuration ---
# IMPORTANT: These must match your training configuration.
MODEL_PATH = 'models/tulu_character_model_best.h5'
TRAIN_DATA_PATH = 'Data/train'  # Path to the training data to build the label map
IMG_HEIGHT = 48
IMG_WIDTH = 48

# --- Caching Functions for Performance ---
@st.cache_resource
def load_trained_model():
    """Loads the trained Keras model from the specified path."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

@st.cache_data
def create_label_map(data_path):
    """Creates a sorted label map from the training data subdirectories."""
    try:
        class_names = sorted(os.listdir(data_path))
        label_map = {i: name for i, name in enumerate(class_names)}
        return label_map
    except Exception as e:
        st.error(f"Error creating label map from path '{data_path}': {e}")
        return None

# --- NEW: Function to create the translation map ---
@st.cache_data
def create_translation_map():
    """
    Creates a forward translation map (folder_name -> Kannada)
    from a reverse map (Kannada -> {folder_names}).
    """
    # This is the user-suggested format, which is great for organization.
    # Note: Ambiguous mappings (like 'a' for both 'ಅ' and 'ಆ') have been removed
    # to ensure each folder name maps to only ONE character.
    kannada_to_folder_map = {
        'ಅ': {'a'},
        'ಆ': {'ā', 'aa'},
        'ಇ': {'i'},
        'ಈ': {'ī', 'ii', 'ee'},
        'ಉ': {'u'},
        'ಊ': {'ū', 'uu', 'oo'},
        'ಋ': {'ṛ', 'ru'},
        'ೠ': {'ṝ', 'ruu'},
        'ಌ': {'ḷ', 'li'},
        'ೡ': {'ḹ', 'lii'},
        'ಎ': {'e'},
        'ಏ': {'ē', 'ee','ae'},
        'ಐ': {'ai', 'aee'},
        'ಒ': {'o'},
        'ಓ': {'ō', 'oo'},
        'ಔ': {'au', 'ou'},
        'ಅಂ': {'aṁ', 'am', 'an', 'aM'},
        'ಅಃ': {'aḥ', 'aha', 'ah'},
        'ಕ': {'ka'},
        'ಖ': {'kha'},
        'ಗ': {'ga'},
        'ಘ': {'gha'},
        'ಙ': {'ṅa', 'nga'},
        'ಚ': {'ca', 'cha'},
        'ಛ': {'chha', 'ccha'},
        'ಜ': {'ja'},
        'ಝ': {'jha'},
        'ಞ': {'ña', 'nya', 'gna'},
        'ಟ': {'ṭa', 'Ta','tta'},
        'ಠ': {'ṭha', 'Tha'},
        'ಡ': {'ḍa', 'Da','Dda'},
        'ಢ': {'ḍha', 'Dha'},
        'ಣ': {'ṇa', 'Na', 'nna'},
        'ತ': {'ta'},
        'ಥ': {'tha'},
        'ದ': {'da'},
        'ಧ': {'dha'},
        'ನ': {'na'},
        'ಪ': {'pa'},
        'ಫ': {'pha', 'fa'},
        'ಬ': {'ba'},
        'ಭ': {'bha'},
        'ಮ': {'ma'},
        'ಯ': {'ya'},
        'ರ': {'ra'},
        'ಲ': {'la'},
        'ವ': {'va', 'wa'},
        'ಶ': {'śa', 'sha', 'sh'},
        'ಷ': {'ṣa', 'Sha', 'ssa', 'ssha'},
        'ಸ': {'sa'},
        'ಹ': {'ha'},
        'ಳ': {'ḷa', 'La', 'lla'},
        'ಕ್ಷ': {'kṣa', 'kSha', 'ksha'},
        'ಜ್ಞ': {'jña', 'gya', 'gn', 'dnya', 'gnya'},
    }

    # Invert the dictionary to create the format the app needs
    folder_name_to_kannada_map = {}
    for kannada_char, folder_names_set in kannada_to_folder_map.items():
        for folder_name in folder_names_set:
            folder_name_to_kannada_map[folder_name] = kannada_char
            
    return folder_name_to_kannada_map

# --- Image Processing and Prediction ---
def preprocess_image(image, size=(IMG_WIDTH, IMG_HEIGHT)):
    """Preprocesses the uploaded image to match the model's input requirements."""
    img_array = np.array(image)
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized_img = cv2.resize(gray_img, size)
    processed_img = resized_img.reshape(1, size[0], size[1], 1) / 255.0
    return processed_img

def predict_character(model, image, label_map):
    """Makes a prediction and returns the label and confidence score."""
    if model is None or label_map is None:
        return "Error", 0.0
    processed_img = preprocess_image(image)
    prediction_probabilities = model.predict(processed_img)
    confidence = np.max(prediction_probabilities)
    predicted_index = np.argmax(prediction_probabilities)
    predicted_label = label_map.get(predicted_index, "Unknown Character")
    return predicted_label, confidence

# --- Main Application ---

# Load the model, label map, and the new translation map
model = load_trained_model()
label_map = create_label_map(TRAIN_DATA_PATH)
folder_name_to_kannada_map = create_translation_map()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Translator", "About"])

if page == "Translator":
    st.title("Tulu Script to Kannada Translator 📜")
    st.markdown("Upload an image of a single Tulu character to see its Kannada equivalent.")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Upload Your Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Tulu Character", use_container_width=True)

    with col2:
        st.subheader("2. Get Translation")
        if uploaded_file:
            if st.button("Translate Character"):
                with st.spinner('Analyzing character...'):
                    tulu_char_folder_name, confidence = predict_character(model, image, label_map)
                    st.markdown("---")
                    st.success(f"**Recognized Tulu Character (Folder):**")
                    st.title(tulu_char_folder_name)
                    st.write(f"**Confidence:** `{confidence:.2%}`")
                    
                    # Use the new map for translation
                    kannada_translation = folder_name_to_kannada_map.get(tulu_char_folder_name, "Translation not available")
                    
                    st.info(f"**Kannada Translation:**")
                    st.title(kannada_translation)
        else:
            st.info("Please upload an image to get started.")

elif page == "About":
    st.title("About This Application")
    st.markdown("""
    This application uses a Convolutional Neural Network (CNN) to recognize handwritten Tulu characters from an image and translate them into their corresponding Kannada characters.
    """)