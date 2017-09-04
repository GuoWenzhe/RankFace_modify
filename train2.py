from __future__ import print_function

from keras.models import Sequential,load_model
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
#import cv2
import sys
import os
import numpy as np
import csv
from PIL import Image
from sklearn.model_selection import train_test_split
import py_face_detection as fd

def shape_of_array(arr):
    array = np.array(arr)
    return array.shape


def make_network():
    model = Sequential()
    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(128, 128, 3)))  
    #model.add(Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    # model.add(Activation('tanh'))

    return model


def main():
    dict = {}
    fd.load_face_location(sys.argv[2], dict)

    x, y, filename = fd.load_image_data(sys.argv[1],dict)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=42)
    x = []
    y = []

    model = make_network()

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    #hist = model.fit(train_x, train_y, batch_size=100, epochs=100, verbose=1)
    #hist = model.fit(train_x, train_y, batch_size=100, nb_epoch=100, verbose=1)
    hist = model.fit(train_x, train_y, batch_size=100,nb_epoch=200,shuffle=True,verbose=1,show_accuracy=True)

    model.evaluate(test_x, test_y, show_accuracy=True)
    scores = model.predict(test_x)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")

    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
 
    # evaluate loaded model on test data
    score = model.evaluate(train_x, train_y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    score = model.predict(train_x)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


if __name__ == '__main__':
    main()
