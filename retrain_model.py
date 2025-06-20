import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io

# === CONFIG ===
IMG_SIZE = 128
DATA_DIRS = ['data', 'user_data']  # train on original + new samples
MODEL_PATH = 'model/bird_model.h5'

# === HELPER: Convert .wav to spectrogram image ===
def audio_to_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    mels_db = librosa.power_to_db(mels, ref=np.max)

    # Convert spectrogram to image in memory
    fig = plt.figure(figsize=(2, 2), dpi=64)
    librosa.display.specshow(mels_db, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    image = Image.open(buf).convert('RGB')
    image = np.array(image) / 255.0  # Normalize
    return image

# === HELPER: Load all spectrograms from both datasets ===
def load_data():
    X = []
    y = []
    label_map = {}
    label_idx = 0

    for base_dir in DATA_DIRS:
        for label in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, label)
            if not os.path.isdir(folder_path):
                continue

            if label not in label_map:
                label_map[label] = label_idx
                label_idx += 1

            for file in os.listdir(folder_path):
                if not file.endswith('.wav'):
                    continue
                try:
                    file_path = os.path.join(folder_path, file)
                    spectrogram = audio_to_spectrogram(file_path)
                    spectrogram = tf.image.resize(spectrogram, [IMG_SIZE, IMG_SIZE])
                    spectrogram = tf.image.rgb_to_grayscale(spectrogram)
                    spectrogram = tf.squeeze(spectrogram).numpy()
                    X.append(spectrogram)
                    y.append(label_map[label])
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = tf.keras.utils.to_categorical(np.array(y), num_classes=len(label_map))
    return X, y, label_map

# === CNN Model ===
def build_model(num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === MAIN TRAINING ===
print(" Loading data...")
X, y, label_map = load_data()
print(f" Loaded {len(X)} samples across {len(label_map)} classes.")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(" Training model...")
model = build_model(num_classes=y.shape[1])
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))

model.save(MODEL_PATH)
print(f" Model saved to: {MODEL_PATH}")
