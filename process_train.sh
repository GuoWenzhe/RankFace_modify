python resize_image.py ./data/train_data/ ./data/train_face/
python face_detection_cv.py ./data/train_face/ list1.txt
/home/hmx/neural-enhance-master/python-3.4/bin/python3 ./train2.py ./data/train_face/ ./list1.txt 
