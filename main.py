#import cv2
import sys
import predict
#import face_detection as fd
import os
import tensorflow as tf
from PIL import Image,ImageDraw
import numpy


def get_face_image(img, faces_coordinate, margin_extend_rate=0.3):
    faces = []
    coordinates = []
    for i in range(len(faces_coordinate)):
        x_str, y_str, w_str, h_str = faces_coordinate[i].strip().split(" ")
        x = int(x_str)
        y = int(y_str)
        w = int(w_str)
        h = int(h_str)
 
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
    return faces,coordinates

if __name__ == '__main__':
    list = sys.argv[1]
    dict = {}
    fp = open(list,"r")
    line = fp.readline()
    while line:
        line = line.strip('\r\n')
        info = line.strip().split("\t")
        value = info[1:]
        dict[info[0]] = value
        line = fp.readline()
    fp.close()

    for i in sys.argv:
        if i.find('.jpg') != -1:
            filename = i
            #img = cv2.imread(filename)
            image = Image.open(filename)
            img = numpy.asarray(image)
            faces, coordinates = get_face_image(img, dict[filename])
            

            #img_drawed = fd.draw_faces(img)

            #font = cv2.FONT_HERSHEY_SIMPLEX
            #faces, coordinates = fd.get_face_image(img)
            for i in range(len(faces)):
                score = predict.predict_cv_img(faces[i])
                score_AQ = predict.get_AQ(score[0][0])
                print(filename, '-', i, ' ', coordinates[i], '  ', score_AQ)
                #drawable.text(str(predict.get_AQ(score[0][0])),  coordinates[i], fill=(255,0,0), font=None)
                #cv2.putText(img_drawed, str(predict.get_AQ(score[0][0])), coordinates[i], font, 0.8, (255, 0, 0), 2)
            #fd.show(img_drawed)
            

