import streamlit as st
from PIL import Image
import numpy as np
import cv2
import re
import unicodedata

# Try to import OCR engines
TESS_AVAILABLE = True
EASYOCR_AVAILABLE = True
try:
    import pytesseract
    from pytesseract import Output
    _ = pytesseract.get_tesseract_version()
except Exception:
    TESS_AVAILABLE = False

try:
    import easyocr
except Exception:
    EASYOCR_AVAILABLE = False

# ---------------- App Config ----------------
st.set_page_config(page_title="Accurate Kannada OCR (Pretrained)", layout="wide")

# ---------------- Helpers ----------------
KANNADA_BLOCK_START = 0x0C80
KANNADA_BLOCK_END = 0x0CFF

def is_kannada_char(ch: str) -> bool:
    """Return True if Unicode char is in Kannada block (U+0C80–U+0CFF)."""
    if not ch:
        return False
    code = ord(ch)
    return KANNADA_BLOCK_START <= code <= KANNADA_BLOCK_END

def filter_to_kannada_or_dash(text: str) -> str:
    """
    Keep Kannada characters; convert everything else (except space) to '-'.
    Collapse multiple '-' and normalize whitespace.
    """
    out = []
    for ch in text:
        if ch.isspace():
            out.append(" ")  # preserve spaces
        elif is_kannada_char(ch):
            out.append(ch)
        else:
            out.append("-")
    result = "".join(out)
    # Collapse repeated dashes and condense whitespace
    result = re.sub(r"-{2,}", "-", result)
    result = re.sub(r"\s{2,}", " ", result)
    # Clean dash around spaces
    result = re.sub(r"\s*-\s*", " - ", result).strip()
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result

def to_cv(image_pil: Image.Image) -> np.ndarray:
    """PIL -> OpenCV BGR"""
    return cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

def deskew_image(img_gray: np.ndarray) -> np.ndarray:
    """
    Estimate skew via Hough transform on edges; rotate to correct it.
    Works best for printed text.
    """
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 200)
    if lines is None:
        return img_gray
    # Compute average angle near 0 or 180°
    angles = []
    for rho, theta in lines[:,0]:
        # Convert to degrees and normalize around 0°
        deg = (theta * 180.0 / np.pi) - 90.0
        if -45 <= deg <= 45:
            angles.append(deg)
    if not angles:
        return img_gray
    angle = np.median(angles)
    (h, w) = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_for_ocr(image_pil: Image.Image, upscale_min_width: int = 1200) -> np.ndarray:
    """
    Robust preprocessing for OCR:
      - convert to gray
      - optional upscale for tiny images
      - deskew
      - denoise + adaptive threshold
    Returns a **binary** OpenCV image suitable for OCR.
    """
    img = to_cv(image_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Upscale small images
    if gray.shape[1] < upscale_min_width:
        scale = upscale_min_width / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Deskew (helps printed docs)
    gray = deskew_image(gray)

    # Denoise & binarize
    gray = cv2.fastNlMeansDenoising(gray, h=15)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Invert if background is black
    white_ratio = (bin_img == 255).mean()
    if white_ratio < 0.5:
        bin_img = 255 - bin_img

    return bin_img

# ---------------- OCR Engines ----------------
@st.cache_resource(show_spinner=False)
def get_easyocr_reader():
    # Kannada code is 'kn' in EasyOCR
    return easyocr.Reader(['kn'], gpu=False)

def ocr_with_tesseract(bin_img: np.ndarray, psm: int = 6) -> str:
    """
    OCR with Tesseract using Kannada LSTM model.
    psm 6 = Assume a single uniform block of text (good general default).
    """
    if not TESS_AVAILABLE:
        return ""
    # Convert to PIL for pytesseract
    pil = Image.fromarray(bin_img)
    config = f'--oem 1 --psm {psm} -l kan'  # LSTM engine, Kannada language
    try:
        text = pytesseract.image_to_string(pil, config=config)
    except Exception:
        text = ""
    return text

def ocr_with_easyocr(image_pil: Image.Image, conf_threshold: float = 0.3) -> str:
    """
    OCR with EasyOCR; returns concatenated text for items >= confidence threshold.
    """
    if not EASYOCR_AVAILABLE:
        return ""
    reader = get_easyocr_reader()
    # EasyOCR expects RGB np array
    results = reader.readtext(np.array(image_pil.convert("RGB")))
    parts = []
    for (bbox, txt, conf) in results:
        if conf >= conf_threshold and txt:
            parts.append(txt)
    return " ".join(parts)

def run_ocr(image_pil: Image.Image, engine: str, psm: int, conf_thr: float) -> tuple[str, np.ndarray]:
    """
    Decide which engine to run and return (raw_text, preprocessed_image).
    engine: 'Auto' | 'Tesseract' | 'EasyOCR'
    """
    pre = preprocess_for_ocr(image_pil)

    if engine == "Tesseract":
        return ocr_with_tesseract(pre, psm), pre

    if engine == "EasyOCR":
        return ocr_with_easyocr(image_pil, conf_thr), pre

    # Auto: prefer Tesseract for printed; fallback to EasyOCR
    text_tess = ocr_with_tesseract(pre, psm) if TESS_AVAILABLE else ""
    text_easy = ocr_with_easyocr(image_pil, conf_thr) if EASYOCR_AVAILABLE else ""

    # Heuristic: choose result with more Kannada characters
    def score_kn(s: str) -> int:
        return sum(1 for ch in s if is_kannada_char(ch))

    if score_kn(text_tess) >= score_kn(text_easy):
        return text_tess, pre
    else:
        return text_easy, pre

# ---------------- UI ----------------
st.title("✅ Accurate Kannada OCR (Pretrained)")
st.caption("Print-focused accuracy with Tesseract (kan) + EasyOCR fallback. Non-Kannada → ‘-’")

with st.sidebar:
    st.header("Settings")
    engine = st.radio("OCR Engine", ["Auto", "Tesseract", "EasyOCR"])

    psm = st.selectbox(
        "Tesseract Page Segmentation (psm)",
        options=[3, 4, 6, 7, 11, 13],
        index=2,
        help=(
            "3: Fully automatic page\n"
            "4: Single column of text\n"
            "6: Single uniform block (default)\n"
            "7: Single text line\n"
            "11: Sparse text\n"
            "13: Raw line"
        ),
    )

    conf_thr = st.slider(
        "EasyOCR Confidence Threshold",
        min_value=0.0, max_value=1.0, value=0.30, step=0.05,
        help="Only EasyOCR detections with confidence ≥ this are kept."
    )

    show_pre = st.checkbox("Show preprocessed image for OCR", value=False)

uploaded = st.file_uploader("📂 Upload an image (printed/handwritten Kannada)", type=["png", "jpg", "jpeg"])

with st.expander("ℹ️ Tesseract setup (Windows)"):
    st.markdown("""
**Steps (one-time):**
1) Install Tesseract OCR: search “Tesseract Windows UB Mannheim” and install.
2) During install, include **Additional language data** and select **Kannada (kan)**.
3) If Streamlit still can’t find Tesseract:
   - Add the install folder (e.g. `C:\\Program Files\\Tesseract-OCR`) to your **PATH**.
   - Or set `pytesseract.pytesseract.tesseract_cmd` to the full exe path in code.

> If Tesseract isn’t available, the app will automatically use EasyOCR.
""")

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("🔍 Run OCR"):
        with st.spinner("Reading…"):
            raw_text, pre_img = run_ocr(image, engine, psm, conf_thr)
            raw_text = unicodedata.normalize("NFC", raw_text or "")

        if show_pre:
            st.subheader("Preprocessed image used for OCR")
            st.image(pre_img, clamp=True, use_container_width=True)

        st.subheader("📝 Raw OCR text")
        st.code(raw_text if raw_text.strip() else "(empty)")

        st.subheader("➡ Kannada-only output (others → '-')")
        filtered = filter_to_kannada_or_dash(raw_text)
        st.title(filtered if filtered else "-")

else:
    st.info("Upload an image to begin.")

# ---------------- Footer note ----------------
if not TESS_AVAILABLE:
    st.warning("Tesseract not detected; using EasyOCR only. For best accuracy on printed text, install Tesseract + Kannada (kan).")
if not EASYOCR_AVAILABLE:
    st.warning("EasyOCR not detected; using Tesseract only.")
