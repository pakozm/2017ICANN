"""
"""
from __future__ import print_function

import sys
import numpy as np
np.random.seed(1337)  # for reproducibility

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping, LambdaCallback
from keras.datasets import mnist
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils
from keras import backend as K

use_fm = True
standardize = True
nb_epoch = 1000 #20000
nb_batch = 200
batch_size = 10000
wdecay = 0.01
input_shape = (2,)
imbalance_proportion = 0.001
class0_samples = 1000000
val_proportion = 0.10

def macro_fm(y_true, y_pred):
    beta = 1.0
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
    beta = 1.0
    beta2 = beta**2.0
    top = K.sum(y_true * y_pred)
    bot = beta2 * K.sum(y_true) + K.sum(y_pred)
    return -(1.0 + beta2) * top / bot

def sample(n, mu, sigma):
    return sigma * np.random.randn(n, 2) + mu

def sample0(n):
    return sample(n, [-10.0, 0.0], [4.0, 4.0])

def sample1(n):
    return sample(n, [2.0, 4.0], [0.5, 0.5])

X_train = np.concatenate([sample0(class0_samples),
                          sample1(int(imbalance_proportion * class0_samples))],
                         axis=0)
Y_train = np.concatenate([np.zeros((class0_samples,1)),
                          np.ones((int(imbalance_proportion * class0_samples),1))],
                         axis=0)

X_val = np.concatenate([sample0(int(class0_samples * val_proportion)),
                        sample1(int(imbalance_proportion * class0_samples * val_proportion))],
                       axis=0)
Y_val = np.concatenate([np.zeros((int(class0_samples * val_proportion),1)),
                        np.ones((int(imbalance_proportion * class0_samples * val_proportion),1))],
                       axis=0)

if standardize:
    X_mean = np.mean(X_train , axis=0)
    X_std = np.std(X_train , axis=0)
    X_std[X_std == 0.0] = 1.0
    X_inv_std=1.0 / X_std

    X_train = (X_train - X_mean) * X_inv_std
    X_val = (X_val - X_mean) * X_inv_std

outputs = inputs = Input(input_shape)
outputs = Dense(1, W_regularizer=l2(wdecay))(outputs)
outputs = Activation('sigmoid')(outputs)
model = Model(input=inputs, output=outputs)
print(model.summary())

print(X_train.shape, X_val.shape, np.sum(Y_train))

if use_fm:
    loss = macro_fm
else:
    loss = 'binary_crossentropy'
model.compile(loss=loss, optimizer='adam',
              metrics=['accuracy', 'fmeasure'])

# show_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[1].get_weights()))
# show_predict = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.predict(X_val)))

for epoch in range(nb_epoch):
    losses = []
    for j in range(nb_batch):
        order = np.random.random_integers(0, X_train.shape[0]-1, batch_size)
        losses.append(model.train_on_batch(X_train[order, :], Y_train[order, :]))
    losses = np.mean(losses, axis=0)
    test_losses = model.test_on_batch(X_val, Y_val)
    print(epoch, losses, test_losses)

#p = model.predict(X_val)
#K.eval(macro_fm(K.variable(Y_val), K.variable(p))), roc_auc_score(Y_val, p))
