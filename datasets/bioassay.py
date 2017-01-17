import os

import numpy as np
import pandas as pd

from keras.utils.data_utils import get_file

URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00209/Schierz_Bioassay.zip'
FILENAME = 'bioassay.zip'

OUTPUT_DICT = {
    'Inactive': 0,
    'Active': 1,
    'Inconc': 0,
    'Inconclusive': 0,
}

DATASETS = [
    'AID1284Morered',
    'AID1284red',
    'AID1608Morered',
    'AID1608red',
    'AID362red',
    'AID373AID439red',
    'AID373red',
    'AID439Morered',
    'AID439red',
    'AID456red',
    'AID604AID644_AllRed',
    'AID604red',
    'AID644Morered',
    'AID644red',
    'AID687AID721red',
    'AID687red',
    'AID688red',
    'AID721morered',
    'AID721red',
    'AID746AID1284red',
    'AID746red',
]
    
def load_data(path=FILENAME, idx=0):
    """Loads the Schierz Bioassay dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
        idx: a number between 0 and 20
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(FILENAME, origin=URL)
    dest = os.tempnam()
    os.system('unzip ' + path + ' -d ' + dest)

    def load_csv(filename):
        path = os.path.join(dest, 'VirtualScreeningData', filename)
        data = pd.read_csv(path).replace({'Outcome': OUTPUT_DICT})
        y_mat = data['Outcome']
        del data['Outcome']
        x_mat = data.values
        order = np.arange(x_mat.shape[0])
        np.random.shuffle(order)
        return x_mat[order], y_mat[order]

    x_train, y_train = load_csv(DATASETS[idx] + '_train.csv')
    x_test, y_test = load_csv(DATASETS[idx] + '_test.csv')

    return (x_train, y_train), (x_test, y_test)
