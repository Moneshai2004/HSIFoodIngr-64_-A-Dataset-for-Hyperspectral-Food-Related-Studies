import os
import numpy as np
from skimage import transform, io
from sklearn.preprocessing import StandardScaler

def augment_image(image):
    # Example augmentation: rotation
    angle = np.random.randint(-30, 30)
    augmented_image = transform.rotate(image, angle)
    return augmented_image

def augment_data(image_folder, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    filenames = os.listdir(image_folder)
    for filename in filenames:
        image = io.imread(os.path.join(image_folder, filename))
        augmented_image = augment_image(image)
        io.imsave(os.path.join(save_folder, f"aug_{filename}"), augmented_image)

if __name__ == "__main__":
    image_folder = 'data/processed'
    save_folder = 'data/augmented'
    augment_data(image_folder, save_folder)
