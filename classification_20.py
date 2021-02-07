import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    GRU,
    LSTM,
    Dropout,
    Flatten,
    Dense,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.applications import (
    MobileNetV2,
    MobileNet,
    InceptionResNetV2,
    Xception,
)
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import os
from keras.preprocessing import image
from numpy import savetxt


def frame_from_video(dir_inp, dir_out):
    files = os.listdir(dir_inp)
    files = sorted(files)
    for f in files:
        print(f)
        vidcap = cv2.VideoCapture(dir_inp + f)
        success, image = vidcap.read()
        count = 0
        while success:
            if count < 10:
                cv2.imwrite(dir_out + f[:11] + "_frame_00%d.jpg" % count, image)
                success, image = vidcap.read()
                count += 1
            else:
                if count < 100:
                    cv2.imwrite(dir_out + f[:11] + "_frame_0%d.jpg" % count, image)
                    success, image = vidcap.read()
                    count += 1
                else:
                    cv2.imwrite(dir_out + f[:11] + "_frame_%d.jpg" % count, image)
                    success, image = vidcap.read()
                    count += 1


def frames_to_array(dir):
    x_data = np.array([])
    y_data = np.array([])
    sum_frames = np.array([])
    img_size = 64
    count_pixels = img_size * img_size
    files = os.listdir(dir)
    files = sorted(files)
    perv = 0
    count = 0
    iter = 0
    for f in files:
        img = image.load_img(dir + f, grayscale=True, target_size=(img_size, img_size))
        x_data = np.append(x_data, img)
        y_data = np.append(y_data, int(f[1:3]) - 1)
        print(f)
        print(x_data.size / count_pixels)
        prev = count
        count = int(f[18:21])
        iter += 1
        if count < prev:
            sum_frames = np.append(sum_frames, iter)
            iter = 0
    return x_data


def frames_count(dir):
    sum_frames = np.array([])
    files = os.listdir(dir)
    files = sorted(files)
    perv = 0
    count = 0
    iter = 0
    for f in files:
        prev = count
        count = int(f[18:21])
        iter += 1
        if count < prev:
            sum_frames = np.append(sum_frames, iter)
            iter = 0
    sum_frames = np.append(sum_frames, 58)
    return sum_frames


def read_csv_frames(dir):
    skip = 0
    max = 4581376

    x_data = np.genfromtxt(
        "all_frames_1channel_128x128_20.csv",
        delimiter=" ",
        skip_header=skip,
        max_rows=max,
    )
    skip += max

    for i in range(99):
        temp_arr = np.genfromtxt(
            "all_frames_1channel_128x128_20.csv",
            delimiter=" ",
            skip_header=skip,
            max_rows=max,
        )
        x_data = np.concatenate((x_data, temp_arr))
        skip += max
        print(i)
    return x_data


def features_from_frames(x_data, dir_out):
    print(x_data.shape)
    x_data_rgb = np.repeat(x_data[..., np.newaxis], 3, -1)
    print(x_data_rgb.shape)
    mbnv2_model = MobileNetV2(
        alpha=1.0,
        weights="imagenet",
        include_top=False,
        input_shape=(64, 64, 3),
        pooling="avg",
    )
    model = tf.keras.Sequential([mbnv2_model])
    X_train_features = model.predict(x_data_rgb)
    print(X_train_features.shape)
    del x_data
    savetxt(dir_out, X_train_features, delimiter=" ")


def stratification(x_data, sum_frames):
    x_data = x_data.reshape(111850, 1280)
    max_len = 142080
    X = np.array([np.zeros([max_len])])
    Y = np.array([])
    n = 0
    for i in range(1000):
        temp = np.array([])
        for j in range(int(sum_frames[i])):
            if n != 111850:
                temp = np.append(temp, x_data[n])
                # print(n)
                n += 1
        temp = temp.reshape(1, temp.shape[0])
        temp = pad_sequences(temp, maxlen=max_len, dtype="float32")
        X = np.append(X, temp)
    X = X.reshape(1001, 111, 1280)
    X = np.delete(X, 0, axis=0)

    for i in range(20):
        for j in range(50):
            Y = np.append(Y, i)

    x_train = np.array([])
    y_train = np.array([])
    for i in range(X.shape[0]):
        if i % 5 != 0:
            x_train = np.append(x_train, X[i])
            y_train = np.append(y_train, Y[i])
    x_train = x_train.reshape(800, 111, 1280)
    x_val = X[::5]
    y_val = Y[::5]
    print(x_val.shape)
    print(y_val.shape)
    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train, x_val, y_val


def read_csv_features():
    x_data = np.array([])
    sum_frames = np.genfromtxt("preproc_lsa20/lsa20_sum_frames.csv", delimiter=" ")
    x_data_0 = np.genfromtxt("preproc_lsa20/lsa20_data_0.csv", delimiter=" ")
    x_data_1 = np.genfromtxt("preproc_lsa20/lsa20_data_1.csv", delimiter=" ")
    x_data_2 = np.genfromtxt("preproc_lsa20/lsa20_data_2.csv", delimiter=" ")
    x_data_3 = np.genfromtxt("preproc_lsa20/lsa20_data_3.csv", delimiter=" ")
    x_data_4 = np.genfromtxt("preproc_lsa20/lsa20_data_4.csv", delimiter=" ")
    x_data_5 = np.genfromtxt("preproc_lsa20/lsa20_data_5.csv", delimiter=" ")
    x_data_6 = np.genfromtxt("preproc_lsa20/lsa20_data_6.csv", delimiter=" ")
    x_data_7 = np.genfromtxt("preproc_lsa20/lsa20_data_7.csv", delimiter=" ")
    x_data_8 = np.genfromtxt("preproc_lsa20/lsa20_data_8.csv", delimiter=" ")
    x_data_9 = np.genfromtxt("preproc_lsa20/lsa20_data_9.csv", delimiter=" ")

    x_data = np.concatenate((x_data_0, x_data_1))
    x_data = np.concatenate((x_data, x_data_2))
    x_data = np.concatenate((x_data, x_data_3))
    x_data = np.concatenate((x_data, x_data_4))
    x_data = np.concatenate((x_data, x_data_5))
    x_data = np.concatenate((x_data, x_data_6))
    x_data = np.concatenate((x_data, x_data_7))
    x_data = np.concatenate((x_data, x_data_8))
    x_data = np.concatenate((x_data, x_data_9))
    return sum_frames, x_data


def rnn(x_data, sum_frames):
    x_train, y_train, x_val, y_val = stratification(x_data, sum_frames)
    y_train = utils.to_categorical(y_train)
    y_val = utils.to_categorical(y_val)

    model = Sequential()
    model.add(LSTM(512, input_shape=(61, 1280)))
    # model.add(Dropout(0.2))
    model.add(Dense(20, activation="softmax"))
    model.summary()
    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    filepath = "pr1/models/model_lsa20_00_04.h5"
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
    )
    learning_rate_reduction = ReduceLROnPlateau(
        monitor="val_accuracy", patience=2, verbose=1, factor=0.5, min_lr=0.00001
    )
    callbacks_list = [checkpoint, learning_rate_reduction]
    history = model.fit(
        x_train,
        y_train,
        epochs=15,
        batch_size=100,
        validation_data=(x_val, y_val),
        callbacks=callbacks_list,
    )
    saved_model = load_model(filepath)
    scores = saved_model.evaluate(x_val, y_val, verbose=0)
    print(scores[1] * 100)