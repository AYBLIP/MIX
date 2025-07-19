import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown

# Definisi fungsi dan kelas untuk efficientneb0
def swish(x):
    return x * tf.nn.sigmoid(x)

class FixedDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

st.title("Klasifikasi Kue dengan Streamlit")

# Pilihan model dan optimizer
model_options = ['MobileNetV2', 'EfficientNetB0', 'NASNetMobile']
optimizer_options = ['Adam', 'SGD', 'RMSprop']
model_choice = st.selectbox("Model", model_options)
optimizer_choice = st.selectbox("Optimizer", optimizer_options)

# Memuat model berdasarkan pilihan
try:
    if model_choice == 'NASNetMobile' and optimizer_choice == 'Adam':
        # Link Google Drive untuk model NASNetMobile dengan optimizer Adam
        gdrive_link = 'https://drive.google.com/file/d/1f4KoGXAed_E14IpOWQllAH_uj0i53EkA/view?usp=drive_link'
        file_id = gdrive_link.split('/d/')[1].split('/')[0]
        model_filename = 'best_model_NASNetMobile_Adam.h5'
        gdown.download(f'https://drive.google.com/uc?id={file_id}', model_filename, quiet=False)
        model = tf.keras.models.load_model(model_filename)
        st.success(f"Model {model_choice} dengan optimizer {optimizer_choice} berhasil dimuat.")
    elif model_choice in ['MobileNetV2', 'NASNetMobile', 'EfficientNetB0']:
        model_path = f'best_model_{model_choice}_{optimizer_choice}.h5'
        if model_choice == 'EfficientNetB0':
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'swish': swish,
                    'FixedDropout': FixedDropout
                }
            )
        else:
            model = tf.keras.models.load_model(model_path)
        st.success(f"Model {model_choice} dengan optimizer {optimizer_choice} berhasil dimuat.")
    else:
        model = None
        st.warning("Model pilihan tidak dikenali.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")

# Daftar kelas
kelas = [
    'Kue Dadar Gulung', 'Kue Kastengel', 'Kue Klepon', 'Kue Lapis',
    'Kue Lumpur', 'Kue Putri Salju', 'Kue Risoles', 'Kue Serabi'
]

# Unggah gambar
uploaded_files = st.file_uploader(
    "Unggah beberapa gambar kue Anda",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    # Membagi layout menjadi dua kolom: satu untuk gambar, satu untuk hasil prediksi
    col1, col2 = st.columns([2, 1])  # Rasio kolom bisa disesuaikan

    with col1:
        # Menampilkan semua gambar secara horizontal
        for uploaded_file in uploaded_files:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            st.image(img, caption=uploaded_file.name, use_container_width=True)

    with col2:
        # Menampilkan prediksi untuk setiap gambar
        for uploaded_file in uploaded_files:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            if model:
                pred = model.predict(img_array)
                pred_kelas = np.argmax(pred, axis=1)[0]
                kelas_terpilih = kelas[pred_kelas]
                confidence = np.max(pred) * 100

                st.write(f"**{uploaded_file.name}**")
                st.write(f"Prediksi: **{kelas_terpilih}**")
                st.write(f"Kepercayaan: {confidence:.2f}%")
            else:
                st.warning("Model belum berhasil dimuat.")
