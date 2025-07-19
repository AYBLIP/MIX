import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown

def swish(x):
    return x * tf.nn.sigmoid(x)

# Definisikan kelas FixedDropout jika digunakan
class FixedDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)
st.title("Klasifikasi Kue dengan Streamlit")

# Pilihan model
model_options = ['MobileNetV2', 'EfficientNetB0', 'NASNetMobile']
model_choice = st.selectbox("Pilih Model", model_options)

# Pilihan optimizer
optimizer_options = ['Adam', 'SGD', 'RMSprop']
optimizer_choice = st.selectbox("Optimizer", optimizer_options)

# Tentukan path model berdasarkan pilihan
import gdown
import tensorflow as tf

# Pastikan fungsi dan variabel yang digunakan seperti swish dan FixedDropout sudah didefinisikan sebelumnya

try:
    if model_choice == 'NASNetMobile' and optimizer_choice == 'Adam':
        gdrive_link = 'https://drive.google.com/file/d/FILE_ID/view?usp=sharing'  # Ganti dengan link nyata
        # Ekstrak file ID dari link
        file_id = gdrive_link.split('/d/')[1].split('/')[0]
        # Nama file lokal
        model_filename = 'NASNetMobile_Adam.keras'
        # Unduh file dari Google Drive
        gdown.download(f'https://drive.google.com/uc?id={file_id}', model_filename, quiet=False)
        model_path = model_filename
    else:
        # Untuk model lain
        if model_choice in ['MobileNetV2', 'NASNetMobile', 'EfficientNetB0']:
            model_path = f'best_model_{model_choice}_{optimizer_choice}.h5'
            # Jika model adalah EfficientNetB0, muat model dengan custom_objects
            if model_choice == 'EfficientNetB0':
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={
                        'swish': swish,
                        'FixedDropout': FixedDropout
                    }
                )
            else:
                # Untuk model lain tanpa custom_objects
                model = tf.keras.models.load_model(model_path)
            st.success(f"Model {model_choice} dengan optimizer {optimizer_choice} berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat model: {str(e)}")
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
