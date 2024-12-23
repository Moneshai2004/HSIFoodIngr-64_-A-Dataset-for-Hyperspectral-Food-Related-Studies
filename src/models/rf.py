from sklearn.ensemble import RandomForestClassifier
import numpy as np

def build_rf_model():
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    return rf_model

def train_rf_model(X_train, y_train):
    rf_model = build_rf_model()
    rf_model.fit(X_train, y_train)
    return rf_model

if __name__ == "__main__":
    # Example: Replace with actual data loading and preprocessing
    X_train = np.random.rand(100, 256*256*3)  # Flattened image features
    y_train = np.random.randint(0, 64, 100)  # Assuming 64 classes for food ingredients

    model = train_rf_model(X_train, y_train)
    print(f"Trained Random Forest model: {model}")
