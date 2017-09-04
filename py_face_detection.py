import sys
import os
import csv
import numpy as np
from PIL import Image


def get_label(num):
    with open('./label.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['#Image'] == str(num):
                return float(row['Attractiveness label'])



def load_face_location(list,dict):
    #list = sys.argv[2]
    fp = open(list,"r")
    line = fp.readline()
    while line:
        line = line.strip('\r\n')
        info = line.strip().split("\t")
        value = info[1:]
        dict[info[0]] = value
        line = fp.readline()
    fp.close()

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

def load_image_data(filedir,dict):
    image_data_list = []
    file_name = []
    label = []
    train_image_list = os.listdir(filedir)
    # train_image_list.remove('.DS_Store')
    for img in train_image_list:
        url = os.path.join(filedir + img)
        # print url
        image = Image.open(url)
        image_np = np.asarray(image)

        faces, coordinates = get_face_image(image_np, dict[img])
        if (len(faces)>1):
            continue
        img_j = Image.fromarray(faces[0])
        img_j = img_j.resize((128,128), Image.ANTIALIAS)
        img_np = np.asarray(img_j)
        image_data_list.append(img_np)
        file_name.append(img)

        img_num = int(img[:img.find('.')])
        att_label = get_label(img_num)
        label.append(att_label)

    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    return img_data,label, file_name

