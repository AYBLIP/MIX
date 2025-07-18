import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Definisikan fungsi swish jika digunakan, misalnya untuk EfficientNetB0
def swish(x):
    return x * tf.nn.sigmoid(x)

# Definisikan kelas FixedDropout jika digunakan
class FixedDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

st.title("Klasifikasi Kue dengan Streamlit")

# Pilihan model
model_options = ['MobileNetV2', 'EfficientNetB0']
model_choice = st.selectbox("Pilih Model", model_options)

# Pilihan optimizer
optimizer_options = ['Adam', 'SGD', 'RMSprop']
optimizer_choice = st.selectbox("Optimizer", optimizer_options)

# Path model
if model_choice == 'MobileNetV2':
    model_path = f'best_model_{model_choice}_{optimizer_choice}.keras'  # Format .keras
else:
    model_path = f'best_model_{model_choice}_{optimizer_choice}.h5'     # Format .h5
# Muat model sesuai pilihan
try:
    if model_choice == 'MobileNetV2':
        # Jika MobileNetV2, muat tanpa custom_objects
        model = tf.keras.models.load_model(model_path)
    else:
        # Jika EfficientNetB0, muat dengan custom_objects
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
