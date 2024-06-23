from PIL import Image
import streamlit as st
from pydub import AudioSegment
from io import BytesIO
import os
import numpy as np
import cv2
import pywt
import tempfile
from scipy.fftpack import dct, idct

# Fungsi untuk melakukan kompresi gambar
def compress_image(image, quality):
    img = image.copy()
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

# Fungsi untuk melakukan kompresi audio
def compress_audio(audio_bytes, bitrate='64k'):
    audio_buf = BytesIO(audio_bytes)
    audio_buf.seek(0) 
    try:
        audio = AudioSegment.from_file(audio_buf, format="mp3")
        compressed_audio_buf = BytesIO()
        audio.export(compressed_audio_buf, format="mp3", bitrate=bitrate)
        return compressed_audio_buf.getvalue()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Fungsi untuk mengompres frame menggunakan DCT
def compress_frame_dct(frame, block_size=8, keep_fraction=0.2):
    frame = np.float32(frame) / 255.0
    frame_dct = np.zeros_like(frame)
    h, w = frame.shape[:2]

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = frame[i:i+block_size, j:j+block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_block[int(block_size*keep_fraction):, int(block_size*keep_fraction):] = 0
            frame_dct[i:i+block_size, j:j+block_size] = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

    frame_compressed = np.uint8(np.clip(frame_dct * 255.0, 0, 255))
    return frame_compressed

# Fungsi untuk mengompres frame menggunakan DWT
def compress_frame_dwt(frame):
    frame = np.float32(frame) / 255.0
    coeffs = pywt.dwt2(frame, 'haar')
    cA, (cH, cV, cD) = coeffs
    
    # Misalnya, menggunakan persentil 95 untuk threshold
    threshold = np.percentile(np.abs(cA), 95)
    cA_thresh = pywt.threshold(cA, threshold, mode='soft')
    
    # Rekonstruksi frame dari koefisien yang telah diproses
    coeffs_thresh = (cA_thresh, (cH, cV, cD))
    frame_idwt = pywt.idwt2(coeffs_thresh, 'haar')
    
    # Pastikan frame_idwt memiliki 3 saluran warna
    if frame_idwt.shape[2] > 3:
        frame_idwt = frame_idwt[:, :, :3]  # Ambil hanya 3 saluran pertama jika lebih dari 3
    
    # Kembalikan frame dalam format yang sesuai
    frame_compressed = np.uint8(np.clip(frame_idwt * 255.0, 0, 255))
    
    return frame_compressed

# Fungsi untuk melakukan kompresi video menggunakan DCT
def compress_video_dct(video_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_in:
            temp_in.write(video_bytes)
            temp_in.close()
            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

            cap = cv2.VideoCapture(temp_in.name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_out, fourcc, cap.get(cv2.CAP_PROP_FPS),
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_compressed = compress_frame_dct(frame)
                out.write(frame_compressed)

            cap.release()
            out.release()

            with open(temp_out, 'rb') as f:
                compressed_video = f.read()

            os.remove(temp_in.name)
            os.remove(temp_out)

            return compressed_video
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Fungsi untuk melakukan kompresi video menggunakan DWT
def compress_video_dwt(video_bytes):
    try:
        # Buat file temporary untuk menyimpan video yang diunggah
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_in:
            temp_in.write(video_bytes)
            temp_in.close()
            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

            # Buka video menggunakan OpenCV
            cap = cv2.VideoCapture(temp_in.name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_out, fourcc, cap.get(cv2.CAP_PROP_FPS),
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Kompresi frame menggunakan DWT
                compressed_frame = compress_frame_dwt(frame)
                
                # Pastikan ukuran frame hasil kompresi sama dengan frame asli
                if compressed_frame.shape != frame.shape:
                    st.warning(f"Ukuran frame tidak sesuai: {frame.shape} vs {compressed_frame.shape}")
                    continue
                
                # Tambahkan frame ke video keluaran
                out.write(compressed_frame)

            cap.release()
            out.release()

            # Baca file video hasil kompresi
            with open(temp_out, 'rb') as f:
                compressed_video = f.read()

            # Hapus file temporary
            os.remove(temp_in.name)
            os.remove(temp_out)

            return compressed_video

    except Exception as e:
        st.error(f"Error: {e}")
        return None
    
# Fungsi untuk menampilkan tombol unduh
def download_button(image_bytes, file_name):
    st.download_button(
        label="Unduh Gambar Kompresi",
        data=image_bytes,
        file_name=file_name,
        mime="image/jpeg"
    )