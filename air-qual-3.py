from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

DATASET_INDEX = 48

MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX]
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

TRAINABLE = True



def generate_model():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))
    # stride = 3
    #
    # x = Permute((2, 1))(ip)
    # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
    #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    x = Masking()(ip)
    x = LSTM(100)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    #out1 = Dense(11,input_dim=11, kernel_initializer='he_uniform', activation='relu')(x)
    out = Dense(1,kernel_initializer='he_uniform')(x)
    model = Model(ip, out)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    # add load model code here to fine-tune

    return model


def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se


if __name__ == "__main__":
    model = generate_model()
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
    estimator = KerasRegressor(build_fn=generate_model, epochs=15, batch_size=5, verbose=1)
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, features, labels, cv=kfold)
    print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    model.save_weights("air-3.h5")
    #train_model(model, DATASET_INDEX, dataset_prefix='air', epochs=600, batch_size=128)

    #evaluate_model(model, DATASET_INDEX, dataset_prefix='air', batch_size=128)
