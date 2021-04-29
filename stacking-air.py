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

DATASET_INDEX = 48
from utils.generic_utils import load_dataset_at, calculate_dataset_metrics, cutoff_choice, \
                                cutoff_sequence
from utils.constants import MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST

MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX]
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]
n_splits = 3
TRAINABLE = True
scores, members = list(), list()
ensemble_scores = list()

def generate_model():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))
    # stride = 10

    # x = Permute((2, 1))(ip)
    # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
    #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    #ip1 = K.reshape(ip,shape=(MAX_TIMESTEPS,MAX_NB_VARIABLES))
    #x = Permute((2, 1))(ip)
    x = Masking()(ip)
    x = AttentionLSTM(8)(x)
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

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

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

def generate_model_2():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))
    # stride = 10

    # x = Permute((2, 1))(ip)
    # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
    #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    #ip1 = K.reshape(ip,shape=(MAX_TIMESTEPS,MAX_NB_VARIABLES))
    #x = Permute((2, 1))(ip)
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))
    # stride = 10

    # x = Permute((2, 1))(ip)
    # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
    #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    #ip1 = K.reshape(ip,shape=(MAX_TIMESTEPS,MAX_NB_VARIABLES))
    #x = Permute((2, 1))(ip)


    x = Masking()(ip)
    x = LSTM(24)(x)
    x = Dropout(0.8)(x)
	
    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = squeeze_excite_block(y)


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


def generate_model_4():
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

def ensemble_predictions(members, testX):
	# make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = array(yhats)
	# sum across ensemble members
    summed = numpy.sum(yhats, axis=0)
    # argmax across classes
    result = argmax(summed, axis=1)
    return result

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
	# select a subset of members
    subset = members[:n_members]
	# make prediction
    yhat = ensemble_predictions(subset, testX)
    # calculate accuracy
    return accuracy_score(testy, yhat)


# fit a stacked model
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
		# make prediction
        yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
        if stackX is None:
           stackX = yhat
        else:
           stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX
 
# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
	# create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
	# fit standalone model
    model = LogisticRegression()
    model.fit(stackedX, inputy)
    return model
 
# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
	# make a prediction
    yhat = model.predict(stackedX)
    return yhat


if __name__ == "__main__":
#train_model(model3, DATASET_INDEX, dataset_prefix='har', epochs=2, batch_size=128)

    #acc3,loss3 = evaluate_model(model3, DATASET_INDEX, dataset_prefix='har', batch_size=128)
    #print('>%.3f' % acc3)
    
    model1 = generate_model()
    
    #train_model(model1, DATASET_INDEX, dataset_prefix='har', epochs=2, batch_size=128)

    #acc, loss=evaluate_model(model1, DATASET_INDEX, dataset_prefix='har', batch_size=128)
    #print('>%.3f' % acc)
    #scores.append(acc)
    model1.load_weights('./weights/air-1.h5')
    members.append(model1)
    model2 = generate_model_2()

    #train_model(model2, DATASET_INDEX, dataset_prefix='har', epochs=2, batch_size=128)

    #acc2,loss2 = evaluate_model(model2, DATASET_INDEX, dataset_prefix='har', batch_size=128)
    #print('>%.3f' % acc2)
    model2.load_weights('./weights/air-2.h5')
    #scores.append(acc2)
    members.append(model2)
    model3 = generate_model_4()
    model3.load_weights('./weights/air-3.h5')
    #scores.append(acc3)
    members.append(model3)
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
    #estimator = KerasRegressor(build_fn=generate_model, epochs=600, batch_size=1, verbose=1)
    #kfold = KFold(n_splits=10)
    #results = cross_val_score(estimator, features, labels, cv=kfold)
    #print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    index = [0]
    #X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
    keys =  list(range(1, X_train.shape[0]))
    keys2 = list(range(X_train.shape[0]+1, X_train.shape[0]*2))
    keys2 = list(range(2*X_train.shape[0]+1, X_train.shape[0]*3))
    print(X_train.shape[0])
    print(model1.predict(X_train).shape)
    
    #df1= pd.DataFrame( {'Model1': [model1.predict(X_train)]}, index= [0])
    #df1 = pd.DataFrame({'lkey': keys,
    #                'value': [model1.predict(X_train)]})
    #print(df1)
    #df2 = pd.DataFrame( {'Model2': [model2.predict(X_train)]}, index= [1])
    #df1 = pd.DataFrame({'lkey': keys,
    #                'value': [model1.predict(X_train)]})
    #print(df2)
    #df3= pd.DataFrame( {'Model3': [model3.predict(X_train)]}, index= [2])
    #print(df3)
    #frames = [df1, df2, df3]
    #print(frames)
    #result = pd.concat(frames)
    #print(result)
    #dummy_data1 = {
    #    'id': [keys],
    #    'Model1': model1.predict(X_train),
    #    'Model2': model2.predict(X_train).ravel(),
#'Model3': model3.predict(X_train),}
    #df1 = pd.DataFrame(dummy_data1, columns = ['Model1', 'Model2','Model3'])
   # print(df1)
    #X_train2 = pd.DataFrame( {'Model1': [model1.predict(X_train),
    # 'Model2': model2.predict(X_train).ravel(),
    # 'Model3': model3.predict(X_train),
    #},index=keys)#).reset_index()
    #print(X_train2)
    out_tr = model1.predict(X_train)
    out_tr2 = model2.predict(X_train)
    out_tr3 = model3.predict(X_train)
    X_train2 = np.column_stack((out_tr,out_tr2))
    X_train2 = np.column_stack((X_train2,out_tr3))
    print(X_train2)
    reg = linear_model.LinearRegression()
    reg.fit(X_train2, y_train)

# prediction using the test set
    out_t = model1.predict(X_test)
    out_t2 = model2.predict(X_test)
    out_t3 = model3.predict(X_test)
    X_test2 = np.column_stack((out_t,out_t2))
    X_test2 = np.column_stack((X_test2,out_t3))
    print(X_test2)
    #reg = linear_model.LinearRegression()
    #reg.fit(X_train2, y_train)
# Don't forget to convert the prediction back to non-log scale
    y_pred = np.exp(reg.predict(X_test2))
    summation = 0
    n = len(y_pred)
    for i in range (0,n):
        difference = y_pred[i] - y_test[i]
        squared_difference = difference**2
        summation = summation + squared_difference
    MSE = summation/n
    RMSE = math.sqrt(MSE)
    print (RMSE)
        