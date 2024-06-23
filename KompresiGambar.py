import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

# Function to compress image using BTC for RGB images
def compress_image_btc(image, block_size=4, quality=50):
    height, width, _ = image.shape
    compressed_image = []

    for channel in cv2.split(image):
        compressed_channel = []
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = channel[i:i+block_size, j:j+block_size]
                block_mean = np.mean(block)
                block_std = np.std(block)

                # Adjust block_mean and block_std based on quality
                block_mean = block_mean * (quality / 100)
                block_std = block_std * (quality / 100)

                # Create a bitmap indicating which pixels are above the mean
                bitmap = block >= (block_mean + block_std)

                # Flatten the bitmap and store it along with the mean and std
                compressed_channel.append((block_mean, block_std, bitmap.flatten().tolist()))
        compressed_image.append(compressed_channel)
    
    return compressed_image

# Function to decompress image using BTC for RGB images
def decompress_image_btc(compressed_image, block_size=4, shape=(0,0)):
    height, width = shape
    decompressed_channels = []
    
    for compressed_channel in compressed_image:
        decompressed_channel = np.zeros((height, width), dtype=np.uint8)
        idx = 0
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                if idx < len(compressed_channel):
                    block_mean, block_std, bitmap = compressed_channel[idx]
                    bitmap = np.array(bitmap)
                    if bitmap.size == block_size * block_size:
                        bitmap = bitmap.reshape((block_size, block_size))
                        block = np.where(bitmap, block_mean + block_std, block_mean - block_std)
                        decompressed_channel[i:i+block_size, j:j+block_size] = block
                idx += 1
        decompressed_channels.append(decompressed_channel)
    
    decompressed_image = cv2.merge(decompressed_channels)
    
    return decompressed_image

# Function to compress image using DCT for each color channel
def dct_compress(image, block_size, quality):
    channels = cv2.split(image)
    dct_channels = []
    for channel in channels:
        height, width = channel.shape
        num_blocks_h = height // block_size
        num_blocks_w = width // block_size
        dct_coeffs = np.zeros((height, width))

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = channel[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size]
                dct_block = cv2.dct(np.float32(block))
                dct_block = np.round(dct_block * (quality / 100.0)) / (quality / 100.0)
                dct_coeffs[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size] = dct_block

        dct_channels.append(dct_coeffs)

    return dct_channels

# Function to decompress image using DCT for each color channel
def dct_decompress(dct_channels, block_size):
    decompressed_channels = []
    for dct_coeffs in dct_channels:
        height, width = dct_coeffs.shape
        image = np.zeros((height, width))
        num_blocks_h = height // block_size
        num_blocks_w = width // block_size

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                dct_block = dct_coeffs[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size]
                block = cv2.idct(np.float32(dct_block))
                image[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size] = block

        image = np.uint8(np.clip(image, 0, 255))
        decompressed_channels.append(image)

    decompressed_image = cv2.merge(decompressed_channels)
    return decompressed_image

# Function to download button
def download_button(data, filename):
    st.download_button(
        label="Unduh Hasil Kompresi",
        data=data,
        file_name=filename,
        mime="image/jpeg"
    )

# Main function for image compression
def kompresi_gambar():
    st.title('Kompresi Gambar')
    st.write("Ini adalah halaman untuk kompresi gambar.")
    st.write("Muat gambar dan kompres dengan kualitas tertentu.")

    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    algorithm_selected = st.selectbox("Pilih Algoritma: ", ["Discrete Cosine Transform (DCT)", "Block Truncation Coding (BTC)"])
    quality = st.slider("Kualitas Kompresi (1-100)", 1, 100, 50)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        if algorithm_selected == "Discrete Cosine Transform (DCT)":
            dct_channels = dct_compress(image_np, block_size=8, quality=quality)
            compressed_image = dct_decompress(dct_channels, block_size=8)
            compressed_image_pil = Image.fromarray(compressed_image)
        elif algorithm_selected == "Block Truncation Coding (BTC)":
            compressed_image = compress_image_btc(image_np, block_size=4, quality=quality)
            decompressed_image = decompress_image_btc(compressed_image, block_size=4, shape=image_np.shape[:2])
            compressed_image_pil = Image.fromarray(decompressed_image.astype(np.uint8))

        st.image(compressed_image_pil, caption=f"Gambar Kompresi dengan {algorithm_selected}")
        
        buffer = BytesIO()
        compressed_image_pil.save(buffer, format="JPEG")
        buffer.seek(0)
        download_button(buffer, "compressed_image.jpg")

if __name__ == "__main__":
    kompresi_gambar()