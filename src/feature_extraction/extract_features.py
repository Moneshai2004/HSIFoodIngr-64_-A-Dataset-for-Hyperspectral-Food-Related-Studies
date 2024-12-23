import numpy as np
from sklearn.decomposition import PCA
from skimage import io
import os

def extract_pca_features(image_folder, n_components=50):
    filenames = os.listdir(image_folder)
    features = []
    
    for filename in filenames:
        image = io.imread(os.path.join(image_folder, filename))
        flattened_image = image.flatten()
        features.append(flattened_image)
    
    features = np.array(features)
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)
    
    return pca_features

if __name__ == "__main__":
    image_folder = 'data/processed'
    features = extract_pca_features(image_folder)
    print(f"Extracted PCA features shape: {features.shape}")
