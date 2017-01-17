'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function

import sys
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils
from keras import backend as K

sys.path.append('.')
from datasets import bioassay as ds_module

def macro_avg_fm_loss(beta=1.0):
    def macro_fm(y_true, y_pred):
        beta2 = beta**2.0
        top = K.sum(y_true * y_pred, axis=0)
        bot = beta2 * K.sum(y_true, axis=0) + K.sum(y_pred, axis=0)
        return -(1.0 + beta2) * K.mean(top/bot)
    return macro_fm

def micro_avg_fm_loss(beta=1.0):
    """
    
    Used with softmax this is equivalent to accuracy
    """
    def micro_fm(y_true, y_pred):
        beta2 = beta**2.0
        top = K.sum(y_true * y_pred)
        bot = beta2 * K.sum(y_true) + K.sum(y_pred)
        return -(1.0 + beta2) * top / bot
    return micro_fm

# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = ds_module.load_data(idx=2)

batch_size = 128
nb_classes = len(np.unique(Y_train))
nb_epoch = 200

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_mean = np.mean(X_train , axis=0)
X_std = np.std(X_train , axis=0)
X_std[X_std == 0.0] = 1.0
X_inv_std=1.0 / X_std

X_train = (X_train - X_mean) * X_inv_std
X_test = (X_test - X_mean) * X_inv_std

print('X_train shape:', X_train.shape)
print('num classes:', nb_classes)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

input_shape = X_train.shape[1:]

# convert class vectors to binary class matrices
if nb_classes > 2:
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

model = Sequential()

# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
#                         border_mode='valid',
#                         input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.25))

# model.add(Flatten())
model.add(Dense(128, input_shape=input_shape, W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128, W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128, W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
if nb_classes > 2:
    model.add(Dense(nb_classes, W_regularizer=l2(0.01)))
    model.add(Activation('softmax'))
else:
    model.add(Dense(1, W_regularizer=l2(0.01)))
    model.add(Activation('sigmoid'))

model.compile(loss=macro_avg_fm_loss(beta=1.0), # 'binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy', 'fbeta_score'])

#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
#                               patience=10, verbose=1, mode='min')

model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, verbose=1, validation_split=0.3)

# callbacks=[early_stopping])
# validation_data=(X_test, Y_test)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test scores:', score)
