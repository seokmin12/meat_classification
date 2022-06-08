import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Dictionary 구성
class_name = ['1++', '1+', '1', '2', '3']
dic = {'1++': 0, '1+': 1, '1': 2, '2': 3, '3': 4}


def find_marbling(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsvLower = np.array([0, 0, 130])  # 추출할 색의 하한(HSV)
    hsvUpper = np.array([140, 160, 190])  # 추출할 색의 상한(HSV)

    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)
    return hsv_mask


def data():
    path = 'train/images/*.jpg'
    X = []
    Y = []

    for image_name in tqdm(glob.glob(path)):
        img = mpimg.imread(image_name)
        label = image_name.split('_')[1]

        marbling_img = find_marbling(img)
        image = cv2.resize(marbling_img, (160, 160))
        X.append(image)
        label = dic[label]
        Y.append(label)

    x_data = np.array(X)
    y_data = np.array(Y)

    x_data = x_data.reshape(len(glob.glob(path)), 160, 160, 1)
    y_data = pd.get_dummies(y_data)
    return x_data, y_data


def create_model():
    X = tf.keras.layers.Input(shape=[160, 160, 1])

    H = tf.keras.layers.Conv2D(16, kernel_size=1, padding='same', activation='swish')(X)
    H = tf.keras.layers.MaxPool2D()(H)

    H = tf.keras.layers.Conv2D(64, kernel_size=5, activation='swish')(H)
    H = tf.keras.layers.MaxPool2D()(H)

    H = tf.keras.layers.Conv2D(256, kernel_size=5, activation='swish')(H)
    H = tf.keras.layers.MaxPool2D()(H)

    H = tf.keras.layers.Conv2D(1024, kernel_size=5, activation='swish')(H)
    H = tf.keras.layers.MaxPool2D()(H)

    H = tf.keras.layers.Conv2D(4096, kernel_size=5, activation='swish')(H)
    H = tf.keras.layers.MaxPool2D()(H)

    H = tf.keras.layers.Conv2D(16384, kernel_size=10, activation='swish')(H)
    H = tf.keras.layers.MaxPool2D()(H)

    H = tf.keras.layers.Flatten()(H)
    H = tf.keras.layers.Dense(10000, activation='swish')(H)
    H = tf.keras.layers.Dense(5000, activation='swish')(H)
    H = tf.keras.layers.Dense(2500, activation='swish')(H)
    H = tf.keras.layers.Dense(1250, activation='swish')(H)
    H = tf.keras.layers.Dense(625, activation='swish')(H)
    H = tf.keras.layers.Dense(320, activation='swish')(H)
    H = tf.keras.layers.Dense(256, activation='swish')(H)
    H = tf.keras.layers.Dense(128, activation='swish')(H)
    H = tf.keras.layers.Dense(120, activation='swish')(H)
    H = tf.keras.layers.Dense(84, activation='swish')(H)
    Y = tf.keras.layers.Dense(5, activation='softmax')(H)

    model = tf.keras.models.Model(X, Y)

    return model


x_data, y_data = data()

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, shuffle=True)

print('학습 데이터:', len(x_train), '테스트 데이터:', len(x_test))

model = create_model()

model.summary()
model.compile(loss='categorical_crossentropy', metrics='accuracy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

model.summary()

history = model.fit(x_train, y_train, epochs=30, batch_size=200, validation_data=(x_test, y_test))
model.save('model/model5.h5')
