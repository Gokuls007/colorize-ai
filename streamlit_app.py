import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
from image_colorizer.inference import colorize_image

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

import os
import datetime
st.sidebar.markdown("---")
st.sidebar.header("📊 Live Model Status")
if os.path.exists(model_path):
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
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original")
        st.image(image, use_container_width=True)
        
    if st.sidebar.button("Colorize"):
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}. Please check the path.")
        else:
            with st.spinner("Colorizing..."):
                # Save temp
                temp_input = "temp_input.jpg"
                image.save(temp_input)
                
                result = colorize_image(
                    temp_input, 
                    model_path, 
                    device=device,
                    saturation_factor=saturation,
                    sharpen_factor=sharpen
                )
                
                with col2:
                    st.header("Colorized")
                    st.image(result, use_container_width=True)
                    
                    # Download button
                    # (Simplified for now)
                    st.success("Colorization Complete!")
else:
    st.info("Please upload an image to start.")
