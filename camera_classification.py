import cv2
import streamlit as st
from PIL import Image

from mqtt_config import send_to_mqtt
import time
import wget
import os

import tensorflow.keras.applications.mobilenet_v2 as mobilenetv2
import tensorflow as tf
import numpy as np

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

# Predict function
def predict(model, cropped_frame):

    frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    
    # Convert frame to PIL image
    img = Image.fromarray(frame_rgb)

    FRAME_WINDOW.image(img.resize((500, 500)))
    
    img = img.resize((224, 224))

    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_confidence = np.max(predictions, axis=1)

    return predicted_class, predicted_confidence

# Streamlit app
st.title("Webcam Live Feed with Trash Classification")
run = st.checkbox('Run')

FRAME_WINDOW = st.image([])

# Threshold confidence
CONFIDENCE_THRESHOLD = 0.80  
CONFIDENCE_THRESHOLD_TO_DB = 0.92   

MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "/sic/kelompok15/conveyor"

model = load_model()
camera = cv2.VideoCapture(1)

predictions_placeholder = st.empty()
message_placeholder = st.empty()

while run:
    _, frame = camera.read()

    # if not _:
    #     st.write("Gagal membaca frame dari kamera.")
    #     st.experimental_rerun()

    height, width, _ = frame.shape
    new_dim = min(height, width)
    top = (height - new_dim) // 2
    left = (width - new_dim) // 2
    cropped_frame = frame[top:top + new_dim, left:left + new_dim]
    
    results, confidence = predict(model, cropped_frame)
    label = results[0]
    confidence = confidence[0]
    
    with predictions_placeholder.container():
        if confidence >= CONFIDENCE_THRESHOLD:
            st.write("Predictions:")
            st.write(f"Class: {label}")
            st.write(f"Confidence: {confidence * 100:.2f}%")

            if confidence >= CONFIDENCE_THRESHOLD_TO_DB:
                send_to_mqtt(MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, f"{label} - {confidence}")

                camera.release()
                
                with message_placeholder:
                    st.write("Data sent to MQTT. Pausing camera for 5 seconds.")
                
                time.sleep(5)
                message_placeholder.empty()
                camera = cv2.VideoCapture(1)
        else:
            st.write("Predictions:")
            st.write("Confidence below threshold, no prediction displayed.")
            st.write("")
    
else:
    st.write('Stopped')
    camera.release()
