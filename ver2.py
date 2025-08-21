import streamlit as st
from PIL import Image
import easyocr
import numpy as np


# --- App Config ---
st.set_page_config(page_title="Tulu Script Translator (Pretrained)", layout="wide")

# --- Initialize Pretrained OCR ---
@st.cache_resource
def load_ocr_model():
    # EasyOCR supports many languages; Kannada is included
    return easyocr.Reader(['kn', 'en'])  # Kannada + English

reader = load_ocr_model()

# --- Simple Translation Mapping (Tulu -> Kannada placeholder) ---
# In real case, you’d need proper mapping between Tulu script & Kannada.
tulu_to_kannada = {
    "a": "ಅ", "aa": "ಆ", "i": "ಇ", "ii": "ಈ",
    "u": "ಉ", "uu": "ಊ", "ka": "ಕ", "kha": "ಖ",
    "ga": "ಗ", "ja": "ಜ", "ta": "ತ", "na": "ನ",
    "ma": "ಮ", "ya": "ಯ", "ra": "ರ", "la": "ಲ",
}

def translate_text(text):
    """Convert OCR result (Tulu/Latin) to Kannada using map."""
    out = ""
    for token in text.split():
        out += tulu_to_kannada.get(token.lower(), token) + " "
    return out.strip()

# --- App UI ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Translator", "About"])

if page == "Translator":
    st.title("📜 Tulu Script Translator (Pretrained EasyOCR)")
    st.write("Upload an image of Tulu/Kannada script and we’ll extract the text using **EasyOCR**.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Translate"):
            with st.spinner("🔍 Running OCR..."):
                results = reader.readtext(np.array(image))

                if not results:
                    st.error("No text detected. Try a clearer image.")
                else:
                    detected_text = " ".join([res[1] for res in results])
                    translated_text = translate_text(detected_text)

                    st.subheader("📝 OCR Detected Text")
                    st.write(detected_text)

                    st.subheader("➡ Kannada Translation")
                    st.title(translated_text if translated_text else "❓ Not Available")

else:
    st.title("ℹ️ About This Pretrained Version")
    st.markdown("""
    This version uses **EasyOCR**, a deep learning OCR system pre-trained on many languages.  

    ✅ No custom `.h5` model needed  
    ✅ Works directly with uploaded images (words/sentences)  
    ✅ Detects Kannada text out-of-the-box  

    ⚠️ Current Tulu → Kannada mapping is **just a placeholder**.  
    To build a true Tulu translator, we need to design a mapping for Tulu script.
    """)
