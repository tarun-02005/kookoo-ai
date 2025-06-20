import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load dataset
data = np.load('bird_data.npz')
X = data['X']
y = data['y']

# Load label names
with open('model/labels.txt', 'r') as f:
    labels = [line.strip() for line in f]

num_classes = len(labels)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))

# Save model
model.save('model/bird_model.h5')
print(f"Model trained and saved with {num_classes} classes.")
