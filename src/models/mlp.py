from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_mlp_model(input_shape=(256*256*3,)):
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape[0], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='softmax'))  # Number of output classes

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = build_mlp_model()
    model.summary()
