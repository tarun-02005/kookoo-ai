import numpy as np
import librosa
import tensorflow as tf

IMG_SIZE = 128
MODEL_PATH = 'model/bird_model.h5'
LABELS_PATH = 'model/labels.txt'

# Load model and labels
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f]

def predict_label(audio_path):
    try:
        y_raw, sr = librosa.load(audio_path, sr=22050, duration=5)
        mel = librosa.feature.melspectrogram(y=y_raw, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = np.resize(mel_db, (IMG_SIZE, IMG_SIZE))
        mel_db = mel_db / 255.0
        input_array = mel_db.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        preds = model.predict(input_array)
        return labels[np.argmax(preds)], float(np.max(preds))
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0