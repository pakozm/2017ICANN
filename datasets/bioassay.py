"""
Number of patterns counts:

     0	     291 /tmp/file5Hus8G/VirtualScreeningData/AID1284Morered_train.csv
     1	     291 /tmp/file5Hus8G/VirtualScreeningData/AID1284red_train.csv
     2	     828 /tmp/file5Hus8G/VirtualScreeningData/AID1608Morered_train.csv
     3	     828 /tmp/file5Hus8G/VirtualScreeningData/AID1608red_train.csv
     4	    3424 /tmp/file5Hus8G/VirtualScreeningData/AID362red_train.csv
     5	   47837 /tmp/file5Hus8G/VirtualScreeningData/AID373AID439red_train.csv
     6	   47832 /tmp/file5Hus8G/VirtualScreeningData/AID373red_train.csv
     7	      57 /tmp/file5Hus8G/VirtualScreeningData/AID439Morered_train.csv
     8	      57 /tmp/file5Hus8G/VirtualScreeningData/AID439red_train.csv
     9	    7987 /tmp/file5Hus8G/VirtualScreeningData/AID456red_train.csv
    10	   47827 /tmp/file5Hus8G/VirtualScreeningData/AID604AID644_AllRed_train.csv
    11	   47832 /tmp/file5Hus8G/VirtualScreeningData/AID604red_train.csv
    12	     166 /tmp/file5Hus8G/VirtualScreeningData/AID644Morered_train.csv
    13	     166 /tmp/file5Hus8G/VirtualScreeningData/AID644red_train.csv
    14	   26455 /tmp/file5Hus8G/VirtualScreeningData/AID687AID721red_train.csv
    15	   26455 /tmp/file5Hus8G/VirtualScreeningData/AID687red_train.csv
    16	   21752 /tmp/file5Hus8G/VirtualScreeningData/AID688red_train.csv
    17	      77 /tmp/file5Hus8G/VirtualScreeningData/AID721morered_train.csv
    18	      77 /tmp/file5Hus8G/VirtualScreeningData/AID721red_train.csv
    19	   47829 /tmp/file5Hus8G/VirtualScreeningData/AID746AID1284red_train.csv
    20	   47832 /tmp/file5Hus8G/VirtualScreeningData/AID746red_train.csv
"""
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
        y_mat = data['Outcome'].values
        del data['Outcome']
        x_mat = data.values
        order = np.arange(x_mat.shape[0])
        np.random.shuffle(order)
        return x_mat[order], y_mat[order]

    x_train, y_train = load_csv(DATASETS[idx] + '_train.csv')
    x_test, y_test = load_csv(DATASETS[idx] + '_test.csv')

    return (x_train, y_train), (x_test, y_test)
