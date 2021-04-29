# stacked generalization with neural net meta model on blobs dataset
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import concatenate, Lambda, Reshape
from keras.optimizers import Adam
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.models import Sequential
from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM
from sklearn.metrics import accuracy_score
#from matplotlib import pyplot
from numpy import mean
from numpy import std
import numpy
from numpy import array
from numpy import argmax
from keras.models import model_from_json, load_model
#import pydot
from keras.utils.vis_utils import model_to_dot
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
from numpy import argmax
from numpy import dstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from numpy import asarray
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from numpy import zeros, newaxis
from sklearn import linear_model
import math
import pandas as pd
import random
from numpy import asarray
from numpy import save


if __name__ == "__main__":
#train_model(model3, DATASET_INDEX, dataset_prefix='har', epochs=2, batch_size=128)

    #acc3,loss3 = evaluate_model(model3, DATASET_INDEX, dataset_prefix='har', batch_size=128)
    #print('>%.3f' % acc3)
    
    air_data = pd.read_excel('AirQualityUCI.xlsx')
    air_data.dropna(axis=0, how='all')
    features = air_data

    features = features.drop('Date', axis=1)
    features = features.drop('Time', axis=1)
    features = features.drop('C6H6(GT)', axis=1)
    features = features.drop('PT08.S4(NO2)', axis=1)

    labels = air_data['C6H6(GT)'].values

    features = features.values
    #X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)
    #X_train=X_train[:, :,newaxis]
    #X = X_train.reshape(1,10,1)
    #y = y_train.reshape(1,10,1)
    features = features[:, :,newaxis]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)
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
        
    save('air-train_X_1_g.npy', X_train)
    
    for i in range (50):
        id1 = random.randrange(0, X_test.shape[0])
        id2 = random.randrange(0, X_test.shape[1])
        id3 = random.randrange(0,X_test.shape[2])
        X_test [id1,id2,id3] = X_test[id1,id2,id3] + s[i]
    
    save('air-test_X_1_g.npy', X_test)
    X_train = []
    X_test = []
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)
    #X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(DATASET_INDEX)
    for i in range (100):
        id1 = random.randrange(0, X_train.shape[0])
        id2 = random.randrange(0, X_train.shape[1])
        id3 = random.randrange(0,X_train.shape[2])
        X_train [id1,id2,id3] = X_train[id1,id2,id3] + s1[i]
        
    save('air-train_X_1_s.npy', X_train)
    
    for i in range (50):
        id1 = random.randrange(0, X_test.shape[0])
        id2 = random.randrange(0, X_test.shape[1])
        id3 = random.randrange(0,X_test.shape[2])
        X_test [id1,id2,id3] = X_test[id1,id2,id3] + s1[i]
    
    save('air-test_X_1_s.npy', X_test)
    
    X_train = []
    X_test = []
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)
    #X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(DATASET_INDEX)
    for i in range (100):
        id1 = random.randrange(0, X_train.shape[0])
        id2 = random.randrange(0, X_train.shape[1])
        id3 = random.randrange(0,X_train.shape[2])
        if np.mod (id1,2) == 0:
             X_train [id1,id2,id3] = X_train[id1,id2,id3] + s1[i]
        else:
             X_train [id1,id2,id3] = X_train[id1,id2,id3] + s[i]
    save('air-train_X_1_c.npy', X_train)
    
    for i in range (50):
        id1 = random.randrange(0, X_test.shape[0])
        id2 = random.randrange(0, X_test.shape[1])
        id3 = random.randrange(0,X_test.shape[2])
        if np.mod (id1,2) == 0:
             X_test [id1,id2,id3] = X_test[id1,id2,id3] + s1[i]
        else:
             X_test [id1,id2,id3] = X_test[id1,id2,id3] + s[i]
    save('air-test_X_1_c.npy', X_test)