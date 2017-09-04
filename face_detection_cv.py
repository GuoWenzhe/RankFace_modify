# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import cv2
#from PIL import Image

INPUT_DIR = './data/train_data/'
def show(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)


def get_face_image(fp, image, margin_extend_rate=0.3):
    img = cv2.imread(sys.argv[1]+image)
    faces = []
    coordinates = []
    faces_coordinate_ = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5)
    )
    fp.write(image)
    for (x, y, w, h) in faces_coordinate_:
        fp.write("\t")
        output_string = str(x) + " " + str(y) + " "+ str(w)+" " + str(h)
        fp.write(output_string)
        
        x_extend = int(w * margin_extend_rate)
        y_extend = int(h * margin_extend_rate)
        if y-y_extend > 0:
            y_min = y-y_extend
        else:
            y_min = 0

        if y+h+y_extend > img.shape[0]:
            y_max = img.shape[0]
        else:
            y_max = y+h+y_extend

        if x-x_extend > 0:
            x_min = x-x_extend
        else:
            x_min = 0

        if x+w+x_extend > img.shape[1]:
            x_max = img.shape[1]
        else:
            x_max = x+w+x_extend

        roi = img[y_min:y_max, x_min:x_max]
        faces.append(roi)
        coordinates.append((x, y))
        # print('FaceDetected')
    fp.write("\n")
    return faces, coordinates


def draw_faces(img):
    faces = []
    image = img
    faces_coordinate_ = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5)
    )
    for (x, y, w, h) in faces_coordinate_:
        #draw = ImageDraw.Draw(image)
        #draw.rectangle((x,y,w,h), fill = (0,255,0))
        cv2.rectangle(image, (x, y), (x + w, y + w), (0, 255, 0), 2)
    cv2.imwrite("result.jpg", img)
    return image

face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')

if __name__ == '__main__':
    list = os.listdir(sys.argv[1])
    fp = open(sys.argv[2], "w")
    for image in list:
       get_face_image(fp,image)
    fp.close()
    #draw_faces(cv2.imread('girls.jpg'))
