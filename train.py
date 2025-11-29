import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------
# CONFIG
# ---------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25   # increased for better accuracy

train_path = "dataset/train"
test_path = "dataset/test"

# ---------------------------
# DATA AUGMENTATION (Improved)
# ---------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5]
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    test_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

num_classes = train_data.num_classes
print("Classes:", train_data.class_indices)

# ---------------------------
# MODEL: MobileNetV2 + Fine-Tuning
# ---------------------------
base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Fine-tune last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

# ---------------------------
# CUSTOM CLASSIFIER HEAD
# ---------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------
# CLASS WEIGHTS (Helps Trash & Plastic)
# ---------------------------
class_weights = {
    0: 1.0,   # cardboard
    1: 1.0,   # glass
    2: 1.2,   # metal
    3: 1.0,   # paper
    4: 1.3,   # plastic
    5: 2.0    # trash (hardest class)
}

# ---------------------------
# TRAIN MODEL
# ---------------------------
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    class_weight=class_weights
)

# ---------------------------
# SAVE MODEL
# ---------------------------
model.save("garbage_classifier.h5")
print("🔥 Improved model saved as garbage_classifier.h5")

# ---------------------------
# ACCURACY / LOSS PLOTS
# ---------------------------
plt.figure()
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Improved Model Accuracy")
plt.legend()
plt.savefig("accuracy_plot.png")

plt.figure()
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Improved Model Loss")
plt.legend()
plt.savefig("loss_plot.png")

print("📊 Saved accuracy_plot.png and loss_plot.png")
