import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("Klasifikasi Kue dengan Streamlit")

# Pilihan model
model_options = ['MobileNetV2', 'EfficientNetB0', 'NASNetMobile']
model_choice = st.selectbox("Pilih Model", model_options)

# Pilihan optimizer
optimizer_options = ['Adam', 'SGD', 'RMSprop']
optimizer_choice = st.selectbox("Optimizer", optimizer_options)

# Tentukan path model berdasarkan pilihan
if model_choice == 'NASNetMobile' and optimizer_choice == 'Adam':
    gdrive_link = 'https://drive.google.com/file/d/1f4KoGXAed_E14IpOWQllAH_uj0i53EkA/view?usp=drive_link'  # Ganti dengan link nyata
    # Ekstrak file ID dari link
    file_id = gdrive_link.split('/d/')[1].split('/')[0]
    # Nama file lokal
    model_filename = 'best_model_NASNetMobile_Adam.h5'
    # Unduh file dari Google Drive
    gdown.download(f'https://drive.google.com/uc?id={file_id}', model_filename, quiet=False)
    model_path = model_filename
else:
    if model_choice in ['MobileNetV2', 'NASNetMobile']:
        model_path = f'model_{model_choice}_{optimizer_choice}.keras'
    else:
        model_path = f'model_{model_choice}_{optimizer_choice}.h5'

# Muat model
try:
    if model_choice in ['MobileNetV2', 'NASNetMobile']:
        model = tf.keras.models.load_model(model_path)
    else:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'swish': swish,
                'FixedDropout': FixedDropout
            }
        )
    st.success(f"Model {model_choice} dengan optimizer {optimizer_choice} berhasil dimuat.")
except Exception as e:
    model = None
    st.error(f"Gagal memuat model dari {model_path}. Error: {str(e)}")
# Daftar kelas
kelas = ['Kue Dadar Gulung', 'Kue Kastengel', 'Kue Klepon', 'Kue Lapis', 'Kue Lumpur', 'Kue Putri Salju', 'Kue Risoles', 'Kue Serabi']

# Unggah beberapa gambar sekaligus
uploaded_files = st.file_uploader("Unggah beberapa gambar kue Anda", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Baca gambar
        img = image.load_img(uploaded_file, target_size=(224, 224))
        st.image(img, caption=uploaded_file.name, use_container_width=1)

        # Pra-pemrosesan gambar
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediksi jika model berhasil dimuat
        if model:
            pred = model.predict(img_array)
            pred_kelas = np.argmax(pred, axis=1)[0]
            kelas_terpilih = kelas[pred_kelas]
            confidence = np.max(pred) * 100

            st.write(f"Prediksi: **{kelas_terpilih}**")
            st.write(f"Kepercayaan: {confidence:.2f}%")
        else:
            st.warning("Model belum berhasil dimuat. Harap pilih model dan optimizer yang sesuai serta pastikan file model tersedia.")
