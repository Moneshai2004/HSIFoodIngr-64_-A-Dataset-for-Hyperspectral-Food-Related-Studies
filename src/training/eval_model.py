from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Load a trained model (if available)
    model = load_model('trained_model.h5')
    X_test = np.random.rand(20, 256, 256, 3)  # Example test data
    y_test = np.random.randint(0, 64, 20)  # Example test labels
    evaluate_model(model, X_test, y_test)
