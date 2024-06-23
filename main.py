import streamlit as st
from Beranda import beranda
from KompresiGambar import kompresi_gambar
from KompresiAudio import kompresi_audio
from KompresiVideo import compress_frame_dct, compress_frame_dwt, compress_video_dct, compress_video_dwt, download_button

# Sidebar navigation function
def sidebar():
    st.set_page_config(page_title='Kompresi Kelompok 1')

    with st.sidebar:
        st.title('Kelompok 1')
        st.write(
            """
            - Ahmad Maulidi Roofiad - 1197050009
            - Haikal Mufid Mubarok - 1217050059
            - Kensa Baeren Deftnamor - 1217050071
            - Kireina Amani Ridiesto - 1217050074
            - Megan Medellin - 1217050078
            """
        )

        st.title('Menu Kompresi')
        options = {
            'Beranda': ' ',
            'Kompresi Gambar': ' ',
            'Kompresi Audio': ' ',
            'Kompresi Video': ' '
        }

        selected_option = st.selectbox('Pilih Menu', list(options.keys()), format_func=lambda x: f'{options[x]} {x}')

    if selected_option == 'Beranda':
        beranda()
    elif selected_option == 'Kompresi Gambar':
        kompresi_gambar()
    elif selected_option == 'Kompresi Audio':
        kompresi_audio()
    elif selected_option == 'Kompresi Video':
        st.title('Kompresi Video')
        st.write("Pilih algoritma dan muat video untuk melakukan kompresi.")

        algorithm_selected = st.selectbox("Algoritma: ", ["Algoritma Discrete Cosine Transform (DCT)", "Algoritma Discrete Wavelet Transform (DWT)"])

        if algorithm_selected == "Algoritma Discrete Cosine Transform (DCT)":
            uploaded_file1 = st.file_uploader("Pilih file video", type=["mp4"], accept_multiple_files=False)

            if uploaded_file1 is not None:
                st.write('File yang diunggah:', uploaded_file1.name)

                if st.button('Kompresi Algoritma DCT'):
                    with st.spinner('Mengompresi video...'):
                        compressed_video = compress_video_dct(uploaded_file1.getvalue())
                    if compressed_video:
                        st.video(compressed_video, format='video/mp4', start_time=0)
                        download_button(compressed_video, "compressed_video_dct.mp4")
                        st.success("Kompresi video menggunakan Algoritma DCT berhasil!")
                    else:
                        st.error("Gagal mengompresi video menggunakan Algoritma DCT.")

        elif algorithm_selected == "Algoritma Discrete Wavelet Transform (DWT)":
            uploaded_file2 = st.file_uploader("Pilih file video", type=["mp4"], accept_multiple_files=False)

            if uploaded_file2 is not None:
                st.write('File yang diunggah:', uploaded_file2.name)

                if st.button('Kompresi Algoritma DWT'):
                    with st.spinner('Mengompresi video...'):
                        compressed_video = compress_video_dwt(uploaded_file2.getvalue())
                    if compressed_video:
                        st.video(compressed_video, format='video/mp4', start_time=0)
                        download_button(compressed_video, "compressed_video_dwt.mp4")
                        st.success("Kompresi video menggunakan Algoritma DWT berhasil!")
                    else:
                        st.error("Gagal mengompresi video menggunakan Algoritma DWT.")

# Run the app
if __name__ == '__main__':
    sidebar()
