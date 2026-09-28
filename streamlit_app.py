import streamlit as st
import torch
from PIL import Image, ImageOps
import os
import datetime
from image_colorizer.inference import _is_lfs_pointer, _load_model, colorize_image


@st.cache_resource(show_spinner=False)
def load_model(path, device, mtime):
    # mtime is part of the cache key so a retrained checkpoint gets reloaded
    return _load_model(path, device)


st.set_page_config(page_title="Image Colorizer AI", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 Image Colorization AI")
st.markdown("### Transform your black & white memories into vibrant colors using Deep Learning.")

st.sidebar.header("⚙️ Settings")
model_path = st.sidebar.text_input("Model Checkpoint", "model_checkpoint.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
saturation = st.sidebar.slider("Saturation Boost", 0.5, 2.0, 1.0)
sharpen = st.sidebar.slider("Sharpening", 0.1, 2.0, 0.8)

st.sidebar.markdown("---")
st.sidebar.header("📊 Live Model Status")
if os.path.exists(model_path) and _is_lfs_pointer(model_path):
    st.sidebar.error("The checkpoint is a Git LFS pointer, not the real weights. "
                     "Run `git lfs pull` to download it.")
elif os.path.exists(model_path):
    mtime = os.path.getmtime(model_path)
    last_updated = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    st.sidebar.info(f"**Last Trained:** {last_updated}")
    st.sidebar.caption("💡 When 'Last Trained' changes, click 'Colorize' again to see the improved results!")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed with ❤️ by **Gokul Sathishkumar**")
else:
    st.sidebar.warning("No model found. Please start training.")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Respect EXIF rotation and normalise PNG/RGBA/palette uploads to RGB
    image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Original")
        st.image(image, width="stretch")

    if st.sidebar.button("Colorize"):
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}. Please check the path.")
        else:
            with st.spinner("Colorizing..."):
                try:
                    model = load_model(model_path, device, os.path.getmtime(model_path))
                    result = colorize_image(
                        image,
                        model_path,
                        device=device,
                        saturation_factor=saturation,
                        sharpen_factor=sharpen,
                        model=model,
                    )
                except Exception as e:
                    st.error(f"Could not colorize the image: {e}")
                    st.stop()

                with col2:
                    st.header("Colorized")
                    st.image(result, width="stretch")

                    # Download button
                    # (Simplified for now)
                    st.success("Colorization Complete!")
else:
    st.info("Please upload an image to start.")
