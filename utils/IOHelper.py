from config import config
import numpy as np
import scipy.io as sio
import os
import pickle
import logging

def get_npz_data(data_dir, sample_size, verbose=True):
    if verbose:
        logging.info('Loading ' + config['all_EEG_file'])
    with np.load(data_dir + config['all_EEG_file']) as f:
        X = f[config['trainX_variable']][:sample_size]
        if verbose:
            logging.info("X training loaded.")
            logging.info(X.shape)
        y = f[config['trainY_variable']][:sample_size]
        if verbose:
            logging.info("y training loaded.")
            logging.info(y.shape)
    return X, y

def get_npz_data_vit(data_dir, verbose=True):
    if verbose:
        logging.info('Loading vit data...')
    with np.load(data_dir) as f:
        X = f['X']
        if verbose:
            logging.info("X training loaded.")
            logging.info(X.shape)
        y = f['y']
        if verbose:
            logging.info("y training loaded.")
            logging.info(y.shape)
    return X, y

def store(x, y, clip=True):
    if clip:
        x = x[:10000]
        y = y[:10000]
    output_x = open('x_clip.pkl', 'wb')
    pickle.dump(x, output_x)
    output_x.close()

    output_y = open('y_clip.pkl', 'wb')
    pickle.dump(y, output_y)
    output_y.close()