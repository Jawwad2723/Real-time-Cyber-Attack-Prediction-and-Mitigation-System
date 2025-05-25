from scapy.all import sniff
from kafka import KafkaProducer
import json

def extract_features(packet):
    features = {
        'Destination Port': packet.dport if packet.haslayer('TCP') or packet.haslayer('UDP') else 0,
        'Flow Duration': packet.time,
        'Fwd Packet Length Max': len(packet),
        'Fwd PSH Flags': int(packet[TCP].flags.PSH) if packet.haslayer('TCP') else 0,
        'Fwd URG Flags': int(packet[TCP].flags.URG) if packet.haslayer('TCP') else 0,
        'Packet Length Mean': len(packet),
        'FIN Flag Count': int(packet[TCP].flags.FIN) if packet.haslayer('TCP') else 0,
        'SYN Flag Count': int(packet[TCP].flags.SYN) if packet.haslayer('TCP') else 0,
        'RST Flag Count': int(packet[TCP].flags.RST) if packet.haslayer('TCP') else 0,
        'PSH Flag Count': int(packet[TCP].flags.PSH) if packet.haslayer('TCP') else 0,
        'ACK Flag Count': int(packet[TCP].flags.ACK) if packet.haslayer('TCP') else 0,
        'URG Flag Count': int(packet[TCP].flags.URG) if packet.haslayer('TCP') else 0,
        'ECE Flag Count': int(packet[TCP].flags.ECE) if packet.haslayer('TCP') else 0,
        'Init_Win_bytes_forward': packet[TCP].window if packet.haslayer('TCP') else 0,
        'min_seg_size_forward': packet[TCP].dataofs if packet.haslayer('TCP') else 0,
    }
    return features

def send_to_kafka(producer, topic, features):
    producer.send(topic, value=features)

def process_packet(packet):
    features = extract_features(packet)
    send_to_kafka(producer, 'network-traffic', features)

if __name__ == "__main__":
    producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    sniff(prn=process_packet, count=10)  # Adjust count or remove for continuous sniffing
