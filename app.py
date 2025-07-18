import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, NASNetMobile
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("Klasifikasi Kue dengan Streamlit")

# Pilihan model
model_options = ['MobileNetV2', 'EfficientNetB0', 'NASNet']
model_choice = st.selectbox("Pilih Model", model_options)

# Pilihan optimizer
optimizer_options = ['Adam', 'SGD', 'RMSprop']
optimizer_choice = st.selectbox("Optimizer", optimizer_options)

# Fungsi untuk memuat model berdasarkan pilihan
def load_model(model_name, optimizer_name):
    input_shape = (224, 224, 3)
    if model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=input_shape)
    elif model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)
    elif model_name == 'NASNet':
        base_model = NASNetMobile(weights=None, include_top=False, input_shape=input_shape)
    else:
        return None

    # Tambahkan classifier head sesuai kebutuhan
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    # Sesuaikan jumlah kelas sesuai dataset Anda
    num_classes = 8
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

    # Muat bobot dari file jika tersedia
    model_path = f'model_{optimizer_choice}.h5'
    try:
        model.load_weights(model_path)
        st.success(f"Model dari {model_path} berhasil dimuat.")
    except:
        st.error(f"Gagal memuat bobot dari {model_path}. Pastikan file tersedia.")

    return model

# Muat model sesuai pilihan
model = load_model(model_choice, optimizer_choice)

# Daftar kelas
kelas = ['Kue Dadar Gulung', 'Kue Kastengel', 'Kue Klepon', 'Kue Lapis', 'Kue Lumpur', 'Kue Putri Salju', 'Kue Risoles', 'Kue Serabi']

# Unggah gambar
uploaded_files = st.file_uploader("Unggah gambar kue Anda", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Baca gambar
        img = image.load_img(uploaded_file, target_size=(224, 224))
        st.image(img, caption=uploaded_file.name, use_container_width=1)

        # Pra-pemrosesan gambar
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediksi
        if model:
            pred = model.predict(img_array)
            pred_kelas = np.argmax(pred, axis=1)[0]
            kelas_terpilih = kelas[pred_kelas]
            confidence = np.max(pred) * 100

            st.write(f"Prediksi: **{kelas_terpilih}**")
            st.write(f"Kepercayaan: {confidence:.2f}%")
        else:
            st.warning("Model belum berhasil dimuat. Harap pilih model dan pastikan file bobot tersedia.")
