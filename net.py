import keras
import tools
from keras import layers as L
from keras.models import Sequential
from keras.utils import plot_model

def data_to_img(data):
    return data

def net11():
    model = Sequential()

    # Inputs are (27,27,103)
    # Conv

    #model.add(L.Lambda(data_to_img, input_shape=(8), output_shape=(103,27,27,1)))
    model.add(L.Conv3D(32,(32,4,4), activation='relu', input_shape=(103,11,11,1))) 

    model.add(L.Conv3D(64,(32,5,5), activation='relu'))
    model.add(L.Dropout(.5))

    model.add(L.Conv3D(128,(32,4,4), activation='relu'))
    model.add(L.Dropout(.5))

    # Fully Connected
    model.add(L.Flatten())
    model.add(L.Dense(128, activation='relu'))
    #model.add(L.Conv2D(1,128,1 activation='relu'))
    model.add(L.Dropout(.5))
    model.add(L.Dense(9, activation='softmax'))
    #model.add(L.Conv2D(1,128,1 activation='softmax'))

    # Loss
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    plot_model(model, show_shapes=True, to_file='model.png')
    return model

def net27():
    model = Sequential()

    # Inputs are (27,27,103)
    # Conv

    #model.add(L.Lambda(data_to_img, input_shape=(8), output_shape=(103,27,27,1)))
    model.add(L.Conv3D(32,(32,4,4), activation='relu', input_shape=(103,27,27,1))) 
    model.add(L.MaxPooling3D(pool_size=(1,2,2)))

    model.add(L.Conv3D(64,(32,5,5), activation='relu'))
    model.add(L.MaxPooling3D(pool_size=(1,2,2)))
    model.add(L.Dropout(.5))

    model.add(L.Conv3D(128,(32,4,4), activation='relu'))
    model.add(L.Dropout(.5))

    # Fully Connected
    model.add(L.Flatten())
    model.add(L.Dense(128, activation='relu'))
    #model.add(L.Conv2D(1,128,1 activation='relu'))
    model.add(L.Dropout(.5))
    model.add(L.Dense(9, activation='softmax'))
    #model.add(L.Conv2D(1,128,1 activation='softmax'))

    # Loss
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    plot_model(model, show_shapes=True, to_file='model.png')
    return model
