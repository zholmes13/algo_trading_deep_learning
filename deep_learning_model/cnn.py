from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


def build_model(X_train):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(MaxPooling1D(strides=4))
    # model.add(Dropout(0.7))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(strides=3))
    # model.add(Dropout(0.7))
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(strides=2))
    # model.add(Dropout(0.7))
    model.add(Flatten())
    model.add(Dense(1000,activation="relu"))
    model.add(Dense(500,activation="relu"))
    model.add(Dense(3,activation="softmax"))
    # model.add(Dense(3,activation="softmax"))
    model.summary()
    return model