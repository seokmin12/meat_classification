import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import cv2
import os
import glob
import json
import pandas as pd
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = tf.keras.models.load_model('model/model4.h5')
class_name = ['1++', '1+', '1', '2', '3']

file = open('./sample_submission.csv')
df = pd.read_csv(file)

id = []
grade = []


def find_marbling(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsvLower = np.array([0, 0, 130])  # 추출할 색의 하한(HSV)
    hsvUpper = np.array([140, 160, 190])  # 추출할 색의 상한(HSV)

    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)
    return hsv_mask


def predict():
    for i in tqdm(range(len(df['id']))):
        image_name = df['id'][i]
        img = mpimg.imread('test/images/' + image_name)

        marbling_img = find_marbling(img)
        canny_img = cv2.resize(marbling_img, (160, 160))

        image = np.array([canny_img])
        image = image.reshape(1, 160, 160, 1)
        predict = model.predict(image)

        id.append(image_name)
        grade.append(class_name[np.argmax(predict)])


predict()

raw_data = {'id': id,
            'grade': grade}
data = pd.DataFrame(raw_data)
data.to_csv('submission.csv', index=False)

