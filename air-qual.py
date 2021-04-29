import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from numpy import asarray
from numpy import save

air_data = pd.read_excel('AirQualityUCI.xlsx')
air_data.dropna(axis=0, how='all')
features = air_data

features = features.drop('Date', axis=1)
features = features.drop('Time', axis=1)
features = features.drop('C6H6(GT)', axis=1)
features = features.drop('PT08.S4(NO2)', axis=1)

labels = air_data['C6H6(GT)'].values

features = features.values
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

print("X_trian shape --> {}".format(X_train.shape))
print("y_train shape --> {}".format(y_train.shape))
print("X_test shape --> {}".format(X_test.shape))
print("y_test shape --> {}".format(y_test.shape))
print((np.unique(y_train)).shape)
print((np.unique(y_test)).shape)

save('train_X.npy', X_train)
save('train_Y.npy', y_train)
save('test_X.npy', X_test)
save('test_Y.npy', y_test)