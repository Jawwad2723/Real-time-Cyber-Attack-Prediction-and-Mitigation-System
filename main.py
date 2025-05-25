import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from google.colab import files

data = pd.read_csv('/content/drive/MyDrive/Attacks.csv')
        
# Load data
X = data.drop(' Label', axis=1)
y = data[' Label']  

# Encode categorical labels (fit on all labels)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)  

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize features using TensorFlow's Normalization layer
normalizer = Normalization(axis=-1)
normalizer.adapt(X_train)

X_train_normalized = normalizer(X_train)
X_test_normalized = normalizer(X_test)

# model architecture
def create_model(learning_rate=0.001, dropout_rate=0.3):
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(X_train_normalized.shape[1],)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(8, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')  
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = create_model(learning_rate=0.001, dropout_rate=0.3)

# Train the model
history = model.fit(X_train_normalized, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Downloading our model
model.save('my_model.h5')
files.download('my_model.h5')

# Evaluate the model on training and test data
train_loss, train_accuracy = model.evaluate(X_train_normalized, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test, verbose=0)

print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
