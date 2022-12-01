import io
import tensorflow as tf
import keras
import h5py
import numpy as np
from PIL import Image
from efficientnet.tfkeras import EfficientNetB4
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

class_names = ['Apple_Apple_Scab',
 'Apple_Black_Rot',
 'Apple_Cedar_Apple_Rust',
 'Apple_Healthy',
 'Bell_Pepper_Bacterial_Spot',
 'Bell_Pepper_Healthy',
 'Blueberry_Healthy',
 'Cherry(Including_Sour)_Healthy',
 'Cherry(Including_Sour)_Powdery_Mildew',
 'Corn(Maize)_Cercospora_Gray_Leaf_Spot',
 'Corn(Maize)_Common_Rust',
 'Corn(Maize)_Healthy',
 'Corn(Maize)_Northern_Leaf_Blight',
 'Cotton_Bacterial_Blight',
 'Cotton_Curl_Virus',
 'Cotton_Fussarium_Wilt',
 'Cotton_Healthy',
 'Grape_Black_Rot',
 'Grape_Esca(Black_Measles)',
 'Grape_Healthy',
 'Grape_Leaf_Blight(Isariopsis_Leaf_Spot)',
 'Objects',
 'Orange_Haunglongbing(Citrus_Greening)',
 'Peach_Bacterial_Spot',
 'Peach_Healthy',
 'Potato_Early_Blight',
 'Potato_Healthy',
 'Potato_Late_Blight',
 'Raspberry_Healthy',
 'Rice_Bacterial_Leaf_Blight',
 'Rice_Blast',
 'Rice_Brown_Spot',
 'Rice_Healthy',
 'Rice_Leaf_Smut',
 'Rice_Sheath_Blight',
 'Rice_Tungro',
 'Soybean_Healthy',
 'Squash_Powdery_Mildew',
 'Strawberry_Healthy',
 'Strawberry_Leaf_Scorch',
 'Sugarcane_Bacterial_Blight',
 'Sugarcane_Healthy',
 'Sugarcane_Red_Rot',
 'Tomato_Bacterial_Spot',
 'Tomato_Early_Blight',
 'Tomato_Healthy',
 'Tomato_Late_Blight',
 'Tomato_Leaf_Mold',
 'Tomato_Mosaic_Virus',
 'Tomato_Septoria_Leaf_Spot',
 'Tomato_Target_Spot',
 'Tomato_Two_Spotted_Spider_Mite',
 'Tomato_Yellow_Leaf_Curl_Virus',
 'Wheat_Brown_Rust',
 'Wheat_Healthy',
 'Wheat_Septoria',
 'Wheat_Stripe_Rust',
 'Wheat_Yellow_Rust']

model = load_model('models/99.6_Disease_Classification_EfficientNet.h5')

def predict_image(img):
    i = load_img(img, target_size=(256, 256))
    i = img_to_array(i) / 255.0
    i = i.reshape(256, 256, 3)
    i = np.expand_dims(i, axis=0)
    x = model.predict(i)[0]
    max_indices = np.argpartition(x, -3)[-3:]
    max_values = x[max_indices]
    first = class_names[max_indices[2]]
    second = class_names[max_indices[1]]
    third = class_names[max_indices[0]]
    print(first, second, third, max_values)
    return first, second, third, max_values
