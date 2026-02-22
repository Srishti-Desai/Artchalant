import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Loads and preprocesses an image for prediction.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid path.")
    
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    return img
