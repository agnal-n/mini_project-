import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

# Parameters
dataset_dir = 'Traffic Sign Recognition/train'  # Change to your dataset path
IMG_SIZE = 32
NUM_CLASSES = 43  # Adjust based on your dataset

# Load images and labels manually
X = []
y = []

for class_id in tqdm(range(NUM_CLASSES), desc="Loading data"):
    class_folder = os.path.join(dataset_dir, str(class_id))
    if not os.path.exists(class_folder):
        print(f"Warning: Folder {class_folder} does not exist.")
        continue
    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        X.append(img)
        y.append(class_id)

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} images.")

# One-hot encode labels
y = tf.keras.utils.to_categorical(y, NUM_CLASSES)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,  # You can increase this if needed
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# Save model
model.save('traffic_sign_model_cv2.h5')
print("Training complete and model saved.")

# Mapping classes to sign names
sign_classes = {
    '0': 'speed limit 20',
    '1': 'speed limit 30',
    '2': 'speed limit 50',
    '3': 'speed limit 60',
    '4': 'speed limit 70',
    '5': 'speed limit 80',
    '6': 'end of speed limit 80',
    '7': 'speed limit 100',
    '8': 'speed limit 120',
    '9': 'no overtaking allowed',
    '10': 'no overtaking for trucks',
    '11': 'T-intersection ahead',
    '12': 'right of way',
    '13': 'give way',
    '14': 'stop',
    '15': 'no entry',
    '16': 'no entry for trucks',
    '17': 'no entry (alternate)',
    '18': 'hazard ahead',
    '19': 'left turn ahead',
    '20': 'right turn ahead',
    '21': 'double curve ahead',
    '22': 'speed breaker ahead',
    '23': 'slippery road',
    '24': 'narrowing road ahead',
    '25': 'work ahead',
    '26': 'traffic lights ahead',
    '27': 'pedestrian crossing',
    '28': 'watch out for children',
    '29': 'caution cyclist',
    '30': 'ice on road',
    '31': 'wild animal crossing',
    '32': 'no entry (duplicate)',
    '33': 'turn right',
    '34': 'turn left',
    '35': 'straight ahead',
    '36': 'right intersection',
    '37': 'Go Straight or Left',
    '38': 'right directional indicator',
    '39': 'left directional indicator',
    '40': 'ring road',
    '41': 'no overtaking',
    '42': 'no truck overtaking' 
}

# Real-time recognition with webcam
model = tf.keras.models.load_model('traffic_sign_model_cv2.h5')

cap = cv2.VideoCapture(0)
print("Starting real-time recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm_img = rgb.astype('float32') / 255.0
    input_img = np.expand_dims(norm_img, axis=0)

    pred = model.predict(input_img)
    class_id = np.argmax(pred)
    confidence = pred[0][class_id]

    sign_name = sign_classes.get(str(class_id), f"Class {class_id}")
    label = f"{sign_name} ({confidence*100:.1f}%)"

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow('Traffic Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
