import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# --- App Configuration ---
st.set_page_config(page_title="Tulu Script Translator", layout="wide")

# --- Constants ---
MODEL_PATH = 'models/tulu_character_model_best.h5'
TRAIN_DATA_PATH = 'Data/train'
IMG_HEIGHT, IMG_WIDTH = 48, 48

# --- Caching Functions ---
@st.cache_resource
def load_trained_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        return None

@st.cache_data
def create_label_map(data_path):
    try:
        class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        return {i: name for i, name in enumerate(class_names)}
    except Exception as e:
        st.error(f"❌ Could not create label map: {e}")
        return None

@st.cache_data
def create_translation_map():
    kannada_to_folder = {
        'ಅ': {'a'}, 'ಆ': {'ā', 'aa'}, 'ಇ': {'i'}, 'ಈ': {'ī', 'ii', 'ee'},
        'ಉ': {'u'}, 'ಊ': {'ū', 'uu', 'oo'}, 'ಋ': {'ṛ', 'ru'}, 'ೠ': {'ṝ', 'ruu'},
        'ಌ': {'ḷ', 'li'}, 'ೡ': {'ḹ', 'lii'}, 'ಎ': {'e'}, 'ಏ': {'ē', 'ee', 'ae'},
        'ಐ': {'ai', 'aee'}, 'ಒ': {'o'}, 'ಓ': {'ō', 'oo'}, 'ಔ': {'au', 'ou'},
        'ಅಂ': {'aṁ', 'am', 'an', 'aM'}, 'ಅಃ': {'aḥ', 'aha', 'ah'},
        'ಕ': {'ka'}, 'ಖ': {'kha'}, 'ಗ': {'ga'}, 'ಘ': {'gha'}, 'ಙ': {'ṅa', 'nga'},
        'ಚ': {'ca', 'cha'}, 'ಛ': {'chha', 'ccha'}, 'ಜ': {'ja'}, 'ಝ': {'jha'}, 'ಞ': {'ña', 'nya', 'gna'},
        'ಟ': {'ṭa', 'Ta', 'tta'}, 'ಠ': {'ṭha', 'Tha'}, 'ಡ': {'ḍa', 'Da', 'Dda'},
        'ಢ': {'ḍha', 'Dha'}, 'ಣ': {'ṇa', 'Na', 'nna'}, 'ತ': {'ta'}, 'ಥ': {'tha'},
        'ದ': {'da'}, 'ಧ': {'dha'}, 'ನ': {'na'}, 'ಪ': {'pa'}, 'ಫ': {'pha', 'fa'},
        'ಬ': {'ba'}, 'ಭ': {'bha'}, 'ಮ': {'ma'}, 'ಯ': {'ya'}, 'ರ': {'ra'},
        'ಲ': {'la'}, 'ವ': {'va', 'wa'}, 'ಶ': {'śa', 'sha', 'sh'},
        'ಷ': {'ṣa', 'Sha', 'ssa', 'ssha'}, 'ಸ': {'sa'}, 'ಹ': {'ha'},
        'ಳ': {'ḷa', 'La', 'lla'}, 'ಕ್ಷ': {'kṣa', 'kSha', 'ksha'},
        'ಜ್ಞ': {'jña', 'gya', 'gn', 'dnya', 'gnya'}
    }
    folder_to_kannada = {f: k for k, folders in kannada_to_folder.items() for f in folders}
    return folder_to_kannada

# --- Image Preprocessing ---
def preprocess_image(image, size=(IMG_WIDTH, IMG_HEIGHT)):
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, size)
    processed = resized.reshape(1, size[0], size[1], 1) / 255.0
    return processed

# --- Character Segmentation ---
def segment_characters(image):
    """Segment characters from an uploaded image (word/line)."""
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_images = []
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:  # filter noise
            char_crop = gray[y:y+h, x:x+w]
            char_img = Image.fromarray(char_crop)
            char_images.append((x, char_img))  # store x for sorting
            boxes.append((x, y, w, h))

    # Sort characters left-to-right
    char_images.sort(key=lambda x: x[0])
    sorted_chars = [img for _, img in char_images]

    return sorted_chars, boxes, thresh

# --- Prediction ---
def predict_character(model, image, label_map, folder_to_kannada):
    processed_img = preprocess_image(image)
    preds = model.predict(processed_img, verbose=0)[0]
    idx = np.argmax(preds)
    folder_name = label_map.get(idx, "Unknown")
    kannada_char = folder_to_kannada.get(folder_name, "❓")
    return kannada_char, folder_name, np.max(preds)

# --- Main App ---
model = load_trained_model()
label_map = create_label_map(TRAIN_DATA_PATH)
folder_to_kannada = create_translation_map()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Translator", "About"])

if page == "Translator":
    st.title("📜 Tulu Script to Kannada Translator")
    st.write("Upload an image of **a word/sentence in Tulu script**, and we’ll segment it into characters and translate.")

    uploaded_file = st.file_uploader("Upload Tulu script image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Translate Image"):
            with st.spinner("🔍 Segmenting and translating..."):
                chars, boxes, thresh = segment_characters(image)

                if not chars:
                    st.error("No characters detected. Try a clearer image.")
                else:
                    translated_text = ""
                    details = []

                    for char_img in chars:
                        kannada_char, folder, conf = predict_character(model, char_img, label_map, folder_to_kannada)
                        translated_text += kannada_char
                        details.append((folder, conf, kannada_char))

                    st.markdown("---")
                    st.subheader("📝 Translation Result")
                    st.title(translated_text)

                    st.subheader("🔎 Character Details")
                    for f, c, k in details:
                        st.write(f"**Tulu folder:** `{f}` | **Conf:** `{c:.2%}` | **Kannada:** {k}")

else:
    st.title("ℹ️ About This Application")
    st.markdown("""
    This app now supports **multi-character OCR translation** 🎉  

    - Segments uploaded Tulu word/sentence images into individual characters  
    - Recognizes each character with a CNN model  
    - Maps them into Kannada script and reconstructs the word  

    Built with **TensorFlow + OpenCV + Streamlit**.
    """)
