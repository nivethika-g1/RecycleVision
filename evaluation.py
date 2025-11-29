import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------
# CONFIG
# -------------------
IMAGE_SIZE = 224
BATCH_SIZE = 32
test_path = "dataset/test"

# -------------------
# TEST DATA GENERATOR
# -------------------
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    test_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False   # DO NOT SHUFFLE → required for confusion matrix
)

# -------------------
# LOAD MODEL
# -------------------
model = load_model("garbage_classifier.h5")

# -------------------
# PREDICT
# -------------------
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)

print("\n================ CLASSIFICATION REPORT ================\n")
print(classification_report(test_data.classes, y_pred, target_names=list(test_data.class_indices.keys())))

print("\n================ CONFUSION MATRIX =====================\n")
print(confusion_matrix(test_data.classes, y_pred))
