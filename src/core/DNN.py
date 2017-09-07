#!/data/pyenv/keras/bin/python

import sys
import os
import numpy as np
import pandas
from keras.models import Sequential
from keras.optimizers import adam
from keras.layers.core import Dense, Activation
from sklearn.preprocessing import StandardScaler, scale
from sklearn.linear_model import Lasso

random_seed = 2
np.random.seed(random_seed)

data_path = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))), 'data')
# data_file = os.path.join(data_path, '57_0811-0821-sold-tbsold-rate.ssv')
data_file = os.path.join(data_path, 'generated.ssv')
model_weights_output = os.path.join(data_path, 'DNN.model')


def gen_data():
    with open(data_file, 'w') as fp:
        fp.write('sold\tclick' + os.linesep)
        x = (np.random.rand(2000) - 1) * 6
        y = np.sin(x)
        y.reshape(-1)
        x.reshape(-1)
        for i in range(y.shape[0]):
            fp.write('%f\t%f%s' % (y[i], x[i], os.linesep))
        fp.flush()
gen_data()

data_frame = pandas.read_table(data_file, delim_whitespace=True, header='infer')
lasso = Lasso(alpha=1)
lasso.fit(data_frame.iloc[:, 1:3], data_frame['sold'])
print(os.linesep, '<Lasso>.coef: ', lasso.coef_)

dataset = data_frame.values
data_input = dataset[:, 1:2]
data_output = dataset[:, 0].reshape(-1, 1)

train_input = data_input[0:1880, :]
train_output = data_output[0:1880, :]
test_input = data_input[1880:, :]
test_output = data_output[1880:, :]

# train_input = scale(train_input)
# scaler = StandardScaler().fit(train_input)
# train_input = scaler.transform(train_input)
# test_input = scaler.transform(test_input)


def build_model():
    model = Sequential()
    model.add(Dense(100, kernel_initializer='uniform', input_dim=1))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    optimizer = adam(lr=0.001)
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model

model = build_model()
model.fit(train_input, train_output, batch_size=10, epochs=100, shuffle=True, verbose=0, validation_split=0.)
predict = model.predict(test_input, batch_size=1)
print(test_output.reshape(1, -1)[0])
print(predict.reshape(1, -1)[0])

train_loss = model.evaluate(train_input, train_output, batch_size=1)
test_loss = model.evaluate(test_input, test_output, batch_size=1)
mse = ((predict - test_output) ** 2).sum() / predict.shape[0]
std = (np.abs(predict - test_output)).sum() / np.abs(test_output - test_output.mean()).sum()
print(os.linesep)
print(os.linesep, 'train score: %.2f' % (train_loss * 100))
print(os.linesep, 'test score:  %.2f' % (test_loss * 100))
print(os.linesep, 'mse: %.2f, std: %.2f' % (mse, std), os.linesep)
sys.exit(0)

# model.save_weights(model_weights_output)


estimater = KerasRegressor(
    build_fn=build_model, epochs=100, batch_size=5, verbose=0,
    validation_split=0.15)
kfold = KFold(n_splits=10, random_state=random_seed)
result = cross_val_score(estimater, data_input, data_output, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (result.mean(), result.std()))

