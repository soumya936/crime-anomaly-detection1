import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

IMG_SIZE = (224,224)

# Load CNN
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base_model.output)
feature_model = Model(base_model.input, x)

scaler = StandardScaler()
pca = PCA(n_components=100)
iso = IsolationForest(contamination=0.05)

def extract_features(img):

    img = img.resize(IMG_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    features = feature_model.predict(img)

    return features

def detect_anomaly(features):

    features = scaler.fit_transform(features)
    features = pca.fit_transform(features)

    pred = iso.fit_predict(features)

    if pred[0] == -1:
        return "Anomaly Detected"
    else:
        return "Normal Activity"
