import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("Klasifikasi Kue dengan Streamlit")

# Pilihan model
# model_options = ['MobileNetV2', 'EfficientNetB0']
# model_choice = st.selectbox("Pilih Model", model_options)

# Pilihan optimizer
optimizer_options = ['Adam', 'SGD', 'RMSprop']
optimizer_choice = st.selectbox("Optimizer", optimizer_options)

# Gabungkan pilihan model dan optimizer untuk menentukan path model
model_path = f'best_model_MobileNetV2_{optimizer_choice}.h5'

# Muat model
try:
    model = tf.keras.models.load_model(
        model_path
    )
    st.success(f"Model {model_choice} dengan optimizer {optimizer_choice} berhasil dimuat.")
except:
    model = None
    st.error(f"Gagal memuat model dari {model_path}. Pastikan file model tersedia.")

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
