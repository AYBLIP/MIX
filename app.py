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
# Membuat kolom untuk dropdown Model dan Optimizer
col1, col2 = st.columns(2)



# Pilihan model dan optimizer
model_options = ['MobileNetV2', 'EfficientNetB0', 'NASNetMobile']
optimizer_options = ['Adam', 'SGD', 'RMSprop']
with col1:
    model_choice = st.selectbox("Model", model_options)
with col2:
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
    # Membagi tampilan ke dalam sejumlah kolom sesuai jumlah gambar
    col_count = min(len(uploaded_files), 4)  # maksimal 4 kolom agar tetap rapi
    cols = st.columns(col_count)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        # Pilih kolom berdasarkan indeks
        col = cols[idx % col_count]
        
        # Baca dan tampilkan gambar di kolom yang dipilih
        img = image.load_img(uploaded_file, target_size=(224, 224))
        col.image(img, caption=uploaded_file.name)

        # Pra-pemrosesan gambar
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediksi jika model sudah dimuat
        if model:
            pred = model.predict(img_array)
            pred_kelas = np.argmax(pred, axis=1)[0]
            kelas_terpilih = kelas[pred_kelas]
            confidence = np.max(pred) * 100

            col.write(f"Prediksi: **{kelas_terpilih}**")
        else:
            col.warning("Model belum berhasil dimuat.")
