import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# -------- Load Data --------
def load_data():
    base_path = "Artchalant"
    images = []
    labels = []

    # AI images
    ai_folder = os.path.join(base_path, "AI images")
    inside_ai = os.listdir(ai_folder)

    if len(inside_ai) == 1 and os.path.isdir(os.path.join(ai_folder, inside_ai[0])):
        ai_folder = os.path.join(ai_folder, inside_ai[0])

    for file in os.listdir(ai_folder):
        img_path = os.path.join(ai_folder, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(1)

    # Human images
    human_folder = os.path.join(base_path, "human images")
    for file in os.listdir(human_folder):
        img_path = os.path.join(human_folder, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(0)

    X = np.array(images) / 255.0
    y = np.array(labels)

    return X, y


# -------- Build Model --------
def build_model():
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -------- Train --------
if _name_ == "_main_":
    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_model()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=4
    )

    model.save("model.h5")
    print("Model training complete and saved as model.h5")
