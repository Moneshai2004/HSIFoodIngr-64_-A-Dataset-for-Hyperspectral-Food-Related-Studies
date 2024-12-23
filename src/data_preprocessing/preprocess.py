import os
import numpy as np
from skimage import io
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_images(image_folder):
    images = []
    filenames = os.listdir(image_folder)
    for filename in filenames:
        img = io.imread(os.path.join(image_folder, filename))
        images.append(img)
    return np.array(images)

def normalize_images(images):
    # Normalize pixel values to range [0, 1]
    return images / 255.0

def preprocess(image_folder, save_folder):
    images = load_images(image_folder)
    images_normalized = normalize_images(images)

    # Save preprocessed images (if needed)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for idx, img in enumerate(images_normalized):
        io.imsave(os.path.join(save_folder, f"preprocessed_{idx}.tiff"), img)

    return images_normalized

if __name__ == "__main__":
    image_folder = 'data/raw'
    save_folder = 'data/processed'
    preprocess(image_folder, save_folder)
