"""
"""
from __future__ import print_function

import sys
import numpy as np
np.random.seed(1337)  # for reproducibility

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils
from keras import backend as K

sys.path.append('.')
from datasets import bioassay as ds_module

def macro_fm(y_true, y_pred, beta=1.0):
    beta2 = beta**2.0
    top = K.sum(y_true * y_pred, axis=0)
    bot = beta2 * K.sum(y_true, axis=0) + K.sum(y_pred, axis=0)
    return -(1.0 + beta2) * K.mean(top/bot)

def micro_fm(y_true, y_pred):
    """Used with softmax this is equivalent to accuracy

    Two other commonly used F measures are the F2, which weighs recall
    higher than precision (by placing more emphasis on false negatives),
    and the F0.5, which weighs recall lower than precision (by
    attenuating the influence of false negatives).

    With beta = 1, this is equivalent to a F-measure. With beta < 1,
    assigning correct classes becomes more important, and with beta >
    1 the metric is instead weighted towards penalizing incorrect
    class assignments.
    """
    beta = 0.1
    beta2 = beta**2.0
    top = K.sum(y_true * y_pred)
    bot = beta2 * K.sum(y_true) + K.sum(y_pred)
    return -(1.0 + beta2) * top / bot


use_fm = True
standardize = True

# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = ds_module.load_data(idx=20)

nb_classes = len(np.unique(Y_train))
nb_epoch = 20000
wdecay = 0.000

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

if standardize:
    X_mean = np.mean(X_train , axis=0)
    X_std = np.std(X_train , axis=0)
    X_std[X_std == 0.0] = 1.0
    X_inv_std=1.0 / X_std

    X_train = (X_train - X_mean) * X_inv_std
    X_test = (X_test - X_mean) * X_inv_std

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('num classes:', nb_classes)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
imbalance = np.sum(Y_train)/float(len(Y_train))
print('imbalance:', imbalance)
batch_size = min(len(Y_train), max(128, int(2.0/imbalance)))
print('batch_size:', batch_size)

input_shape = X_train.shape[1:]

# convert class vectors to binary class matrices
if nb_classes > 2:
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

outputs = inputs = Input(input_shape)
outputs = Dense(128, input_shape=input_shape, W_regularizer=l2(wdecay))(outputs)
outputs = Activation('relu')(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(128, W_regularizer=l2(wdecay))(outputs)
outputs = Activation('relu')(outputs)
outputs = Dropout(0.5)(outputs)
#outputs = Dense(128, W_regularizer=l2(wdecay))(outputs)
#outputs = Activation('relu')(outputs)
#outputs = Dropout(0.5)(outputs)
if nb_classes > 2:
    ouptuts = Dense(nb_classes, W_regularizer=l2(wdecay))(outputs)
    outputs = Activation('softmax')(outputs)
else:
    outputs = Dense(1, W_regularizer=l2(wdecay))(outputs)
    outputs = Activation('sigmoid')(outputs)
model = Model(input=inputs, output=outputs)

print(model.summary())

if use_fm:
    loss = micro_fm
else:
    loss = 'binary_crossentropy'
model.compile(loss=loss, optimizer='adam',
              metrics=['accuracy', 'fbeta_score'])

#model.save("/tmp/jarl.mdl")
#del model
#model = load_model("/tmp/jarl.mdl",
#                   custom_objects={"macro_fm": macro_fm,
#                                   "micro_fm": micro_fm})

#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
#                               patience=10, verbose=1, mode='min')

#x_tr, x_va, y_tr, y_va = train_test_split(X_train, Y_train,
#                                          random_state=87654321,
#                                          stratify=Y_train,
#                                          test_size=0.3)

model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, verbose=1, validation_split=0.3)

# num_batches = 10
# for i in range(nb_epoch):
#     tr_losses = []
#     for j in range(num_batches):
#         batch_idx = np.random.randint(0, len(x_tr), batch_size)
#         x_train_batch = x_tr[batch_idx, :]
#         y_train_batch = y_tr[batch_idx]
#         tr_losses.append(model.train_on_batch(x_train_batch,
#                                               y_train_batch))
#     tr_loss = np.mean(tr_losses, axis=0)
#     va_loss = model.evaluate(x_va, y_va, verbose=2)
#     print(i, tr_loss.tolist(), 'val', va_loss)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test scores:', score)
