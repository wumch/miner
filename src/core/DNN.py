#!/data/pyenv/keras/bin/python

import sys
import os
import numpy as np
import pandas
from keras.models import Sequential
from keras.optimizers import adam
from keras.layers.core import Dense, Activation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


def gen_data():
    with open(data_file, 'w') as fp:
        fp.write('sold\tclick\ttbsold' + os.linesep)
        y = np.random.rand(200) * 3.78
        x1 = np.sin(y)
        x2 = np.random.standard_normal(200)
        y.reshape(-1)
        x1.reshape(-1)
        x2.reshape(-1)
        for i in range(y.shape[0]):
            fp.write('%f\t%f\t%f%s' % (y[i], x1[i], x2[i], os.linesep))
        fp.flush()
# gen_data()


data_path = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))), 'data')
data_file = os.path.join(data_path, 't121_ctr_tbsold.tsv')
model_weights_output = os.path.join(data_path, 't121_ctr_tbsold.model')

data_frame = pandas.read_table(data_file, delim_whitespace=True, header='infer')
lasso = Lasso(alpha=1)
lasso.fit(data_frame.iloc[:, [5, 6]], data_frame['ctr'])
print(os.linesep, '<Lasso>.coef: ', lasso.coef_)

dataset = data_frame.values
train_size = 7000
data_input = dataset[:, [5, 6]]
data_output = dataset[:, 0].reshape(-1, 1)
data_input = StandardScaler().fit(data_input).transform(data_input)
data_output = StandardScaler().fit(data_output).transform(data_output)
train_input = data_input[0:train_size, :]
train_output = data_output[0:train_size, :]
test_input = data_input[train_size:, :]
test_output = data_output[train_size:, :]


def build_model():
    model = Sequential()
    model.add(Dense(50, input_dim=2))
    model.add(Activation('relu'))
    model.add(Dense(30))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    optimizer = adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

random_seed = 2
np.random.seed(random_seed)

model = build_model()
model.fit(train_input, train_output, batch_size=10, epochs=200, verbose=0, validation_split=0.2)
model.save_weights(model_weights_output)
predict = model.predict(test_input, batch_size=1)
print(test_output.reshape(1, -1)[0])
print(predict.reshape(1, -1)[0])

train_score = model.evaluate(train_input, train_output, batch_size=1)
test_score = model.evaluate(test_input, test_output, batch_size=1)
mse = ((predict - test_output) ** 2).sum() / predict.shape[0]
std = (np.abs(predict - test_output)).sum() / np.abs(test_output - test_output.mean()).sum()
print(os.linesep)
print(os.linesep, 'train score: %.2f' % (train_score * 100))
print(os.linesep, 'test score:  %.2f' % (test_score * 100))
print(os.linesep, 'mse: %.2f, std: %.2f' % (mse, std), os.linesep)
sys.exit(0)

# model.save_weights(model_weights_output)


estimater = KerasRegressor(
    build_fn=build_model, epochs=100, batch_size=5, verbose=0,
    validation_split=0.15)
kfold = KFold(n_splits=10, random_state=random_seed)
result = cross_val_score(estimater, data_input, data_output, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (result.mean(), result.std()))

