import tensorflow as tf
import os

# --- Constants ---
DATASET_PATH = "D:\my projects\scrip_translator\Data\set"   # Path to your dataset
IMG_HEIGHT, IMG_WIDTH = 48, 48
BATCH_SIZE = 32
EPOCHS = 15

# --- Load Dataset ---
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",   # characters are usually grayscale
    batch_size=BATCH_SIZE
)

# Save class names BEFORE mapping
class_names = train_ds.class_names
num_classes = len(class_names)
print("✅ Found classes:", class_names)

# Normalize images (0–1 range)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# --- Build Model ---
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --- Train Model ---
history = model.fit(train_ds, epochs=EPOCHS)

# --- Save Model ---
os.makedirs("models", exist_ok=True)
model.save("models/tulu_character_model_best1.h5")

print("✅ Model training complete and saved at models/tulu_character_model_best1.h5")
