import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.data_preprocessing.preprocess import preprocess
from src.feature_extraction.extract_features import extract_features
from src.models.cnn import build_cnn_model
from src.training.train_model import train_and_evaluate_model
from src.utils.utils import save_results

def main():
    # Step 1: Load and preprocess data
    raw_data_folder = 'data/raw'
    processed_data_folder = 'data/processed'
    print("Preprocessing the data...")
    processed_images = preprocess(raw_data_folder, processed_data_folder)
    
    # Step 2: Feature extraction
    print("Extracting features from the images...")
    features = extract_features(processed_images)
    
    # Step 3: Split the data into training and testing sets
    print("Splitting the data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Step 4: Build the model (CNN in this case)
    print("Building the CNN model...")
    model = build_cnn_model(input_shape=X_train.shape[1:])
    
    # Step 5: Train and evaluate the model
    print("Training the model...")
    train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Step 6: Save the results
    results = {
        'model': model,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }
    save_results(results, 'results/model_results.pkl')
    print("Results saved successfully.")

if __name__ == "__main__":
    main()
