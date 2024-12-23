from src.models.cnn import build_cnn_model
from src.models.rf import train_rf_model
from src.models.mlp import build_mlp_model
from sklearn.model_selection import train_test_split
import numpy as np

def train_model(X, y, model_type='cnn'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'cnn':
        model = build_cnn_model(input_shape=X_train.shape[1:])
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    elif model_type == 'rf':
        model = train_rf_model(X_train, y_train)
        print(f"Random Forest model score: {model.score(X_test, y_test)}")
    elif model_type == 'mlp':
        model = build_mlp_model(input_shape=(X_train.shape[1],))
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

if __name__ == "__main__":
    X = np.random.rand(100, 256*256*3)  # Example feature data
    y = np.random.randint(0, 64, 100)  # Example labels
    train_model(X, y, model_type='cnn')
