import numpy as np
import pywt
import wave
import struct
from io import BytesIO
from scipy.fftpack import dct, idct
from pydub import AudioSegment
import tempfile
import os
import streamlit as st

# Fungsi untuk membaca file audio
def read_audio(file_bytes):
    try:
        with wave.open(BytesIO(file_bytes), 'rb') as wav_file:
            params = wav_file.getparams()
            frames = wav_file.readframes(params.nframes)
            audio = np.frombuffer(frames, dtype=np.int16)
        return audio, params
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

# Fungsi untuk menulis file audio
def write_audio(audio, params, format='wav'):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
            with wave.open(temp_wav_file.name, 'wb') as wav_file:
                wav_file.setparams(params)
                frames = struct.pack("%dh" % len(audio), *audio)
                wav_file.writeframes(frames)

            # Load temporary WAV file and export to MP3 using pydub
            audio_segment = AudioSegment.from_wav(temp_wav_file.name)
            mp3_filename = temp_wav_file.name.replace('.wav', '.mp3')
            audio_segment.export(mp3_filename, format='mp3')

        # Read the MP3 file back as bytes
        with open(mp3_filename, 'rb') as mp3_file:
            mp3_bytes = mp3_file.read()

        # Remove temporary files
        os.remove(temp_wav_file.name)
        os.remove(mp3_filename)

        return mp3_bytes
    except FileNotFoundError:
        st.error("ffmpeg tidak ditemukan. Pastikan ffmpeg sudah diinstal dan ditambahkan ke PATH.")
        return None

# Fungsi untuk kompresi audio menggunakan DWT
def dwt_compress(audio, level=1):
    coeffs = pywt.wavedec(audio, 'db1', level=level)
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs = list(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))
    return coeffs, threshold

# Fungsi untuk dekompresi audio menggunakan DWT
def dwt_decompress(coeffs):
    audio_reconstructed = pywt.waverec(coeffs, 'db1')
    audio_reconstructed = np.clip(audio_reconstructed, -32768, 32767)
    audio_reconstructed = np.array(audio_reconstructed, dtype=np.int16)
    return audio_reconstructed

# Fungsi untuk kompresi audio menggunakan DCT
def dct_compress(audio):
    audio = audio - np.mean(audio)
    dct_audio = dct(audio, norm='ortho')
    threshold = np.median(np.abs(dct_audio)) / 0.6745
    dct_audio[np.abs(dct_audio) < threshold] = 0
    compressed_audio = idct(dct_audio, norm='ortho')
    compressed_audio = np.clip(compressed_audio, -32768, 32767)
    return np.array(compressed_audio, dtype=np.int16), threshold

# Fungsi untuk menampilkan tombol unduh
def download_button(data_bytes, file_name, mime_type):
    st.download_button(
        label=f"Unduh {file_name}",
        data=data_bytes,
        file_name=file_name,
        mime=mime_type
    )

# Fungsi utama kompresi audio
def kompresi_audio():
    st.title('Kompresi Audio')
    st.write("Unggah file audio (WAV) dan kompres dengan Discrete Wavelet Transform (DWT) atau Discrete Cosine Transform (DCT).")

    uploaded_file = st.file_uploader("Pilih file audio", type=["wav"], accept_multiple_files=False)

    if uploaded_file is not None:
        st.write('File yang diunggah:', uploaded_file.name)

        audio, params = read_audio(uploaded_file.read())

        if audio is None or params is None:
            return

        algorithm = st.selectbox("Pilih Algoritma Kompresi:", ["DWT", "DCT"])

        if st.button('Kompresi'):
            if algorithm == "DWT":
                coeffs, threshold = dwt_compress(audio)
                compressed_audio = dwt_decompress(coeffs)
                file_name = "compressed_audio_dwt.wav"
                st.write("DWT Threshold: ", threshold)
            elif algorithm == "DCT":
                compressed_audio, threshold = dct_compress(audio)
                file_name = "compressed_audio_dct.wav"
                st.write("DCT Threshold: ", threshold)

            compressed_audio_bytes = write_audio(compressed_audio, params, format='mp3')
            if compressed_audio_bytes:
                st.audio(compressed_audio_bytes, format='audio/mp3', start_time=0)

                st.download_button(
                    label="Unduh Audio Kompresi",
                    data=compressed_audio_bytes,
                    file_name=file_name.replace('.wav', '.mp3'),
                    mime="audio/mp3"
                )
                st.success(f"Kompresi audio berhasil! File disimpan sebagai {file_name.replace('.wav', '.mp3')}")
