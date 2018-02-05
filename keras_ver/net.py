import keras
from keras import layers as L
from keras.models import Sequential

def net():
    model = Sequential()

    # Inputs are (27,27,103)
    # Conv
    model.add(L.Conv3D(32,32,4,4, activation='relu', input_shape=(103,27,27,1)))
    model.add(L.MaxPooling3D(pool_size=(1,2,2)))

    model.add(L.Conv3D(64,32,5,5, activation='relu'))
    model.add(L.MaxPooling3D(pool_size=(1,2,2)))
    model.add(L.Dropout(.5))

    model.add(L.Conv3D(128,32,4,4, activation='relu'))
    model.add(L.Dropout(.5))

    # Fully Connected
    model.add(L.Flatten())
    model.add(L.Dense(128, activation='relu'))
    model.add(L.Dropout(.5))
    model.add(L.Dense(10, activation='softmax'))

    # Loss
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model