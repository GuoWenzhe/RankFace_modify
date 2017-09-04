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
import py_face_detection as fd

def get_percentage(score):
    for i in range(len(list)):
        if score < float(list[i]):
            return (i + 1.0) / 500.0

def get_AQ(score):
    score = float(score)
    percentage = get_percentage(score)
    z_score = norm.ppf(percentage)
    return int(100 + (z_score * 24))


def main():
    dict = {}
    fd.load_face_location(sys.argv[2], dict)

    x,label,file_name = fd.load_image_data(sys.argv[1],dict)

    # load weights into new model
    model = load_model('faceRank.h5')
    model.load_weights("model.h5")
    print("Loaded model from disk")
 
    score = model.predict(x)
    score = score * 5.0
    for i in range(len(score)):
        print(file_name[i],' ', label[i], ' ', score[i])
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


if __name__ == '__main__':
    main()
