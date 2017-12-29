import keras
from keras import layers as L
from keras.models import Sequential

def net():
    model = Sequential()

    # Inputs are (27,27,103)

    # Conv
    model.add(L.Conv3D(32, (32,4,4), strides=(32,1,1), activation='relu', input_shape=(103,27,27)))
    model.add(L.MaxPooling2D())

    model.add(L.Conv3D(64, (32,5,5), strides=(32,1,1), activation='relu'))
    model.add(L.MaxPooling2D())
    model.add(L.Dropout(.5))

    model.add(L.Conv3D(128, (32,4,4), strides=(32,1,1), activation='relu'))
    model.add(L.Dropout(.5))

    # Fully Connected
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))

    # Loss
    model.compile(optimizer='adagrad',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    return model