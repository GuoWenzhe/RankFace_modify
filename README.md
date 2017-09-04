# RankFace_modify
this project was modify by this version: https://github.com/Entropy-xcy/RankFace

## Installation

```shell
apt-get install python-dev python-pip -y
git clone https://github.com/Entropy-xcy/RankFace
cd ./RankFace
pip install -r requirements.txt
apt-get install python-opencv
```

the requirement are as follows:

```shell
keras
cv2/PIL(I use PIL，因为python3没装上opencv，自己测试时间太紧，要是有opencv就不用将face_detection单独拆分出来记录到list.txt中了……）
numpy 
sklearn
```

label.csv 手动给每一张图片打分
#training steps:

```shell
python resize_image.py ./data/train_data/ ./data/train_face/       #将train_data进行缩放，
python face_detection_cv.py ./data/train_face/ list1.txt           #将train_face中的脸部定位进行记录，输出到文件保存
/home/hmx/neural-enhance-master/python-3.4/bin/python3 ./train2.py ./data/train_face/ ./list1.txt    #使用keras对模型进行训练
```

#testing steps:

```shell
python resize_image.py ./data/test_data/ ./data/test_face/
python face_detection_cv.py ./data/test_face/ ./list1.txt
/home/hmx/neural-enhance-master/python-3.4/bin/python3 test2.py ./data/train_face/ ./list1.txt  #load_model，并进行输出
```
#the mode and images are in baiduDISK, 
http://pan.baidu.com/s/1boKItEf   password: bkii
