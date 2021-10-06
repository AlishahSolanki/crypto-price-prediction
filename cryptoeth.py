import tensorflow as tf
import numpy as np
import json
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
endpoint = 'https://min-api.cryptocompare.com/data/v2/histoday'
response = requests.get(endpoint + '?fsym=ETH&tsym=USD&limit=1999&api_key=346a5653477eed39d369f3b05523c27a427a45ab31125be8a8dbf06cf4478d49')
dataTable = pd.DataFrame(json.loads(response.content)['Data']['Data'])
dataTable = dataTable.set_index('time')
dataTable.index = pd.to_datetime(dataTable.index, unit='s')
target_col = 'close'
dataTable.drop(["conversionType", "conversionSymbol"], axis = 'columns', inplace = True)
dataTable.head(5)
dataTable.tail(5)
def data_split(dt, test_size=0.2, validation_size=0.25):
    remaining, test = train_test_split(dt, test_size=test_size, shuffle=False)
    train,validation  = train_test_split(remaining, test_size=validation_size, shuffle=False) # 0.25 x 0.8 = 0.2
    return train, validation, test
train, validation, test  = data_split(dataTable, test_size=0.2, validation_size=0.25)
def line_plot(line1, line2, line3, label1=None, label2=None, label3=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.plot(line3, label=label3, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    line_plot(train[target_col], validation[target_col], test[target_col], 'training', 'validation', 'test')
def normalise_zero_base(dt):
    return dt / dt.iloc[0] - 1

def normalise_min_max(dt):
    return (dt - dt.min()) / (dt.max() - dt.min())
def extract_window_data(dt, window_len=5, zero_base=True):
    window_data = []

    for idx in range(len(dt) - window_len):
        tmp = dt[idx: (idx + window_len)].copy()

 
        print("idx",idx, tmp.values,tmp)

        window_data.append(tmp.values)
    return np.array(window_data)
def prepare_data(dt, target_col, window_len=10, zero_base=True, test_size=0.2, validation_size=0.25):
    
    train_d, validation_d, test_d = data_split(dt, test_size=test_size,validation_size=validation_size)
#     print(test_d)
    X_train = extract_window_data(train_d, window_len, zero_base)
    X_validation = extract_window_data(validation_d, window_len, zero_base)
    X_test = extract_window_data(test_d, window_len, zero_base)
#     print("X_test",X_test)
    y_train = train_d[target_col][window_len:].values
    y_validation = validation_d[target_col][window_len:].values
    y_test = test_d[target_col][window_len:].values
#     print("y_test",y_test)
    if zero_base:
        y_train = y_train / train_d[target_col][:-window_len].values - 1
        y_validation = y_validation / validation_d[target_col][:-window_len].values - 1
        y_test = y_test / test_d[target_col][:-window_len].values - 1

    return train_d, validation_d, test_d, X_train, X_validation, X_test, y_train, y_validation, y_test
def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear',
                     dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

np.random.seed(42)
window_len = 5
test_size = 0.2
validation_size=0.25
zero_base = True
lstm_neurons = 100
epochs = 20
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'

train_d, validation_d, test_d, X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(
    dataTable, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size, validation_size=validation_size)

model = build_lstm_model(X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss, optimizer=optimizer)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'],'r',linewidth=2, label='Train loss')
plt.plot(history.history['val_loss'], 'g',linewidth=2, label='Validation loss')
plt.title('LSTM')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()
history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

results = model.evaluate(X_test, y_test, batch_size=batch_size)
print("test loss, test acc:", results)

targets = test[target_col][window_len:]
preds = model.predict(X_test).squeeze()
mean_absolute_error(preds, y_test)

from sklearn.metrics import mean_squared_error
MAE=mean_squared_error(preds, y_test)
MAE


from sklearn.metrics import r2_score
R2=r2_score(y_test, preds)
R2

preds = test[target_col].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
fig, ax = plt.subplots(1, figsize=(13, 7))
ax.plot(targets, label='actual', linewidth=3)
ax.plot(preds, label='prediction', linewidth=3)
ax.set_ylabel('price [USD]', fontsize=14)
ax.set_title("", fontsize=16)
ax.legend(loc='best', fontsize=16)

  <matplotlib.legend.Legend object at 0x1c81b6b7340>
  plt.show()