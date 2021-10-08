import tensorflow as tf
import json
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class Currency():
    target_col = 'close'
    endpoint = 'https://min-api.cryptocompare.com/data/v2/histoday'
    key = '346a5653477eed39d369f3b05523c27a427a45ab31125be8a8dbf06cf4478d49'
    currency = 'BTC'

    np.random.seed(42)
    window_len = 5
    test_size = 0.2
    validation_size = 0.25
    zero_base = True
    lstm_neurons = 100
    epochs = 20
    batch_size = 32
    loss = 'mse'
    dropout = 0.2
    optimizer = 'adam'

    # def __init__(self, currency):
    #     self.dataTable = self.getData(self,currency)

    def getData(self, currency) -> object:
        response = requests.get(self.endpoint + '?fsym=' + currency + '&tsym=USD&limit=1999&api_key=' + self.key)
        print('response',response)
        dataTable = pd.DataFrame(json.loads(response.content)['Data']['Data'])
        dataTable = dataTable.set_index('time')
        dataTable.index = pd.to_datetime(dataTable.index, unit='s')
        dataTable.drop(["conversionType", "conversionSymbol"], axis='columns', inplace=True)
        # self.dataTable = dataTable
        targets, preds = self.callAlgo(dataTable)
        return targets, preds

    def data_split(self, dt, test_size=0.2, validation_size=0.25):
        remaining, test = train_test_split(dt, test_size=test_size, shuffle=False)
        train, validation = train_test_split(remaining, test_size=validation_size, shuffle=False)  # 0.25 x 0.8 = 0.2
        return train, validation, test

    def extract_window_data(self, dt, window_len=5, zero_base=True):
        window_data = []
        for idx in range(len(dt) - window_len):
            tmp = dt[idx: (idx + window_len)].copy()
            window_data.append(tmp.values)
        return np.array(window_data)

    def prepare_data(self, dt):
        train_d, validation_d, test_d = self.data_split(dt, self.test_size, self.validation_size)

        X_train = self.extract_window_data(train_d, self.window_len, self.zero_base)
        X_validation = self.extract_window_data(validation_d, self.window_len, self.zero_base)
        X_test = self.extract_window_data(test_d, self.window_len, self.zero_base)
        print("train_d",train_d)
        y_train = train_d[self.target_col][self.window_len:].values
        y_validation = validation_d[self.target_col][self.window_len:].values
        y_test = test_d[self.target_col][self.window_len:].values
        if self.zero_base:
            y_train = y_train / train_d[self.target_col][:-self.window_len].values - 1
            y_validation = y_validation / validation_d[self.target_col][:-self.window_len].values - 1
            y_test = y_test / test_d[self.target_col][:-self.window_len].values - 1

        return train_d, validation_d, test_d, X_train, X_validation, X_test, y_train, y_validation, y_test

    def build_lstm_model(self, input_data, output_size, neurons=100, activ_func='linear', dropout=0.2, loss='mse',
                         optimizer='adam'):
        model = Sequential()
        model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
        model.add(Dropout(dropout))
        model.add(Dense(units=output_size))
        model.add(Activation(activ_func))
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def callAlgo(self, dataTable):
        train_d, validation_d, test_d, X_train, X_validation, X_test, y_train, y_validation, y_test = self.prepare_data(dataTable)

        model = self.build_lstm_model(X_train, output_size=1, neurons=self.lstm_neurons, dropout=self.dropout,
                                      loss=self.loss,
                                      optimizer=self.optimizer)

        history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=self.epochs,
                            batch_size=self.batch_size, verbose=1, shuffle=True)
        results = model.evaluate(X_test, y_test, batch_size=self.batch_size)

        targets = test_d[self.target_col][self.window_len:]
        preds = model.predict(X_test).squeeze()
        mean_absolute_error(preds, y_test)

        MAE = mean_squared_error(preds, y_test)

        R2 = r2_score(y_test, preds)

        preds = test_d[self.target_col].values[:-self.window_len] * (preds + 1)
        preds = pd.Series(index=targets.index, data=preds)
        print("targets", targets)
        print("preds", preds)
        return targets, preds
