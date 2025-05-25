import json
import numpy as np
import tensorflow as tf
from kafka import KafkaConsumer
from tensorflow.keras.models import load_model

ModelPath="D:\CEP\kafka\my_model (2).h5"
model = load_model('ModelPath')

#configration
consumer = KafkaConsumer(
    'network_data',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='consumer_group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)
def preprocess_data(data):
    processed_data = np.array([
        data['Destination Port'],
        data['Flow Duration'],
        data['Fwd Packet Length Max'],
        data['Fwd PSH Flags'],
        data['Fwd URG Flags'],
        data['Packet Length Mean'],
        data['FIN Flag Count'],
        data['SYN Flag Count'],
        data['RST Flag Count'],
        data['PSH Flag Count'],
        data['ACK Flag Count'],
        data['URG Flag Count'],
        data['ECE Flag Count'],
        data['Init_Win_bytes_forward'],
        data['min_seg_size_forward']
    ]).reshape(1, -1)
    
    normalized_data = tf.keras.utils.normalize(processed_data)
    print(f"Preprocessed and normalized data: {normalized_data}")
    return normalized_data

no_attack_detected = True

#consuming messages
for message in consumer:
    data = message.value
    print(f"Received data: {data}")
    if data:
        no_attack_detected = False
    if no_attack_detected:
        print("No attack is detected")
    else:
        processed_data = preprocess_data(data)
        prediction = model.predict(processed_data)
        predicted_class = np.argmax(prediction, axis=1)
        print(f"Prediction: {predicted_class}")
