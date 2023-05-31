import math
import numpy as np

def split(ids, train, val, test):
    # proportions of train, val, test
    assert (train+val+test == 1)
    IDs = np.unique(ids)
    num_ids = len(IDs)

    # priority given to the test/val sets
    test_split = math.ceil(test * num_ids)
    val_split = math.ceil(val * num_ids)
    train_split = num_ids - val_split - test_split

    train = np.where(np.isin(ids, IDs[:train_split]))[0]
    val = np.where(np.isin(ids, IDs[train_split:train_split+val_split]))[0]
    test = np.where(np.isin(ids, IDs[train_split+val_split:]))[0]
    
    return train, val, test