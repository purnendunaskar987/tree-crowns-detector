import streamlit as st
from deepforest import main
import rasterio
import pandas as pd
import requests

# Set page layout and title
st.set_page_config(page_title="Tree Crown Detection", page_icon="🌲", layout="wide")

# Custom CSS for colorful buttons and upload section
st.markdown(
    """
    <style>
    /* Styling the Detect Trees button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1b5e20, #43a047) !important;
        color: white !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        box-shadow: 0 4px 14px rgba(67, 160, 71, 0.35) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #2e7d32, #66bb6a) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(67, 160, 71, 0.45) !important;
        color: white !important;
    }

    /* Styling the File Uploader dropzone and Browse button */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #43a047 !important;
        background-color: rgba(67, 160, 71, 0.04) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #0288d1, #26c6da) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(2, 136, 209, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: linear-gradient(135deg, #0277bd, #00acc1) !important;
        transform: scale(1.02) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 1. Initialize the Pre-trained AI
@st.cache_resource
def load_model():
    model = main.deepforest()
    model.load_model("weecology/deepforest-tree")
    return model

model = load_model()

st.title("Tree Crown Detection & Canopy Estimator")

# 2. Input Selection
input_method = st.radio("How do you want to provide the image?", ["Upload a File", "Paste a Web Link"])

image_path = None
is_tiff = False
manual_resolution = 0.5

if input_method == "Upload a File":
    uploaded_file = st.file_uploader("Upload Image (.tif, .jpg, .png, .jpeg)", type=["tif", "tiff", "jpg", "jpeg", "png"])
    if uploaded_file is not None:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext in ['tif', 'tiff']:
            is_tiff = True
            
        image_path = f"temp_image.{ext}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if not is_tiff:
            st.info("Since you uploaded a standard image (JPG/PNG), the AI needs to know how big a pixel is in the real world.")
            manual_resolution = st.number_input("Meters per pixel (leave as 0.5 if unsure):", value=0.5, step=0.1)

elif input_method == "Paste a Web Link":
    url = st.text_input(
        "Paste image URL below(e.g., https://example.com/forest.jpg)",
        placeholder="https://example.com/forest.jpg"
    )
    if url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            image_path = "temp_url_image.jpg"
            with open(image_path, "wb") as f:
                f.write(response.content)
            st.success("Image downloaded successfully!")
            
            st.info("Since you are using a web link, the AI needs to know how big a pixel is in the real world.")
            manual_resolution = st.number_input("Meters per pixel (leave as 0.5 if unsure):", value=0.5, step=0.1)
            
        except Exception as e:
            st.error(f"Could not download image. Error: {e}")

# 3. Process Image
if image_path and st.button("Detect Trees"):
    with st.spinner('Detecting trees... this might take a moment.'):
        boxes = model.predict_image(path=image_path)
        
        if boxes is None or len(boxes) == 0:
            st.warning("No trees detected in this image.")
        else:
            pixel_size_x, pixel_size_y = manual_resolution, manual_resolution
            
            if is_tiff:
                try:
                    with rasterio.open(image_path) as src:
                        pixel_size_x, pixel_size_y = src.res
                        st.success(f"Automatically pulled real-world scale from TIFF: {pixel_size_x:.2f} meters/pixel")
                except:
                    st.info("Could not read TIFF metadata. Using default 0.5 resolution.")
                    pixel_size_x, pixel_size_y = 0.5, 0.5
            
            pixel_area_m2 = pixel_size_x * pixel_size_y
            tree_count = len(boxes)
            
            boxes['width'] = boxes['xmax'] - boxes['xmin']
            boxes['height'] = boxes['ymax'] - boxes['ymin']
            boxes['area_m2'] = boxes['width'] * boxes['height'] * pixel_area_m2
            
            total_canopy_area = boxes['area_m2'].sum()
            
            col1, col2 = st.columns(2)
            col1.metric("Total Trees Detected", f"{tree_count:,}")
            col2.metric("Est. Canopy Area (sq meters)", f"{total_canopy_area:,.2f}")
            
            st.subheader("Raw Detection Data")
            st.dataframe(boxes[['xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'area_m2']].head())

# 4. Footer pushed further down
st.markdown(
    """
    <div style='text-align: center; margin-top: 120px; padding-bottom: 30px;'>
        <hr style='border: none; border-top: 1px solid #e0e0e0; margin-bottom: 20px;'>
        <p style='color: #757575; font-size: 14px; font-weight: 500;'>Made with 🍃 and ❤️ for our forests</p>
    </div>
    """,
    unsafe_allow_html=True
)