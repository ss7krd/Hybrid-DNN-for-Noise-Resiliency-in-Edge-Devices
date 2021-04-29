import numpy as np
from numpy import mean
from numpy import std
import numpy
from numpy import array
from numpy import argmax
from numpy import argmax
from numpy import dstack
import random
from numpy import asarray
from numpy import save

DATASET_INDEX = 11
from utils.generic_utils import load_dataset_at, calculate_dataset_metrics, cutoff_choice, \
                                cutoff_sequence
from utils.constants import MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST

#MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX]
#MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
#NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]
#n_splits = 3


if __name__ == "__main__":
#train_model(model3, DATASET_INDEX, dataset_prefix='har', epochs=2, batch_size=128)

    #acc3,loss3 = evaluate_model(model3, DATASET_INDEX, dataset_prefix='har', batch_size=128)
    #print('>%.3f' % acc3)
    
    
    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(DATASET_INDEX)
    mean = np.mean(X_train)
    std  = np.std(X_train)
    print("X_trian shape --> {}".format(X_train.shape))
    print("y_train shape --> {}".format(y_train.shape))
    print("X_test shape --> {}".format(X_test.shape))
    print("y_test shape --> {}".format(y_test.shape))
    print(mean)
    print(std)
    a = 0.7
    s = np.random.normal(mean, a*std, 1000)
    s1 = np.random.poisson(mean, 1000)
    print(X_train.shape[0])
    print(X_train.shape[1])
    print(X_train.shape[2])
    for i in range (100):
        id1 = random.randrange(0, X_train.shape[0])
        id2 = random.randrange(0, X_train.shape[1])
        id3 = random.randrange(0,X_train.shape[2])
        X_train [id1,id2,id3] = X_train[id1,id2,id3] + s[i]
        
    save('train_X_1_g.npy', X_train)
    
    for i in range (50):
        id1 = random.randrange(0, X_test.shape[0])
        id2 = random.randrange(0, X_test.shape[1])
        id3 = random.randrange(0,X_test.shape[2])
        X_test [id1,id2,id3] = X_test[id1,id2,id3] + s[i]
    
    save('test_X_1_g.npy', X_test)
    X_train = []
    X_test = []
    
    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(DATASET_INDEX)
    for i in range (100):
        id1 = random.randrange(0, X_train.shape[0])
        id2 = random.randrange(0, X_train.shape[1])
        id3 = random.randrange(0,X_train.shape[2])
        X_train [id1,id2,id3] = X_train[id1,id2,id3] + s1[i]
        
    save('train_X_1_s.npy', X_train)
    
    for i in range (50):
        id1 = random.randrange(0, X_test.shape[0])
        id2 = random.randrange(0, X_test.shape[1])
        id3 = random.randrange(0,X_test.shape[2])
        X_test [id1,id2,id3] = X_test[id1,id2,id3] + s1[i]
    
    save('test_X_1_s.npy', X_test)
    
    X_train = []
    X_test = []
    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(DATASET_INDEX)
    for i in range (100):
        id1 = random.randrange(0, X_train.shape[0])
        id2 = random.randrange(0, X_train.shape[1])
        id3 = random.randrange(0,X_train.shape[2])
        if np.mod (id1,2) == 0:
             X_train [id1,id2,id3] = X_train[id1,id2,id3] + s1[i]
        else:
             X_train [id1,id2,id3] = X_train[id1,id2,id3] + s[i]
    save('train_X_1_c.npy', X_train)
    
    for i in range (50):
        id1 = random.randrange(0, X_test.shape[0])
        id2 = random.randrange(0, X_test.shape[1])
        id3 = random.randrange(0,X_test.shape[2])
        if np.mod (id1,2) == 0:
             X_test [id1,id2,id3] = X_test[id1,id2,id3] + s1[i]
        else:
             X_test [id1,id2,id3] = X_test[id1,id2,id3] + s[i]
    save('test_X_1_c.npy', X_test)