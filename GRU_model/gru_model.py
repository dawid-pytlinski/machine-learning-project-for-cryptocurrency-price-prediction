import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, GRU

# Pobieranie danych
stock_symbol = 'BTC-USD'
data = yf.download(tickers=stock_symbol, period='5y', interval='1d')
opn = data[['Open']]

# Przygotowanie danych
ds = opn.values
norm = MinMaxScaler(feature_range=(0, 1))
ds_scaled = norm.fit_transform(np.array(ds).reshape(-1, 1))

# Funkcja do budowania zestawu danych
def build_ds(dataset, step):
    X, Y = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        X.append(a)
        Y.append(dataset[i + step, 0])
    return np.array(X), np.array(Y)

# Podział na dane treningowe i testowe
train_size = int(len(ds_scaled) * 0.70)
test_size = len(ds_scaled) - train_size
ds_train, ds_test = ds_scaled[0:train_size, :], ds_scaled[train_size:len(ds_scaled), :1]

# Przygotowanie danych do modelu
time_stamp = 100
Xtrain, Ytrain = build_ds(ds_train, time_stamp)
Xtest, Ytest = build_ds(ds_test, time_stamp)
Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], 1)
Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], 1)

# Budowanie modelu
model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(Xtrain.shape[1], 1)))
model.add(GRU(units=50, return_sequences=True))
model.add(GRU(units=50))
model.add(Dense(units=1, activation='linear'))

# Kompilacja i trenowanie modelu
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=100, batch_size=64)

# Predykcja na danych treningowych i testowych
train_predict = model.predict(Xtrain)
test_predict = model.predict(Xtest)
train_predict = norm.inverse_transform(train_predict)
test_predict = norm.inverse_transform(test_predict)

# Rysowanie wykresów
plt.figure(figsize=(12,6))
plt.plot(ds, label='Original Data')
plt.plot(np.arange(len(ds) - test_size, len(ds) - test_size + len(test_predict)), test_predict.flatten(), label='Predicted Data', color='orange') 
plt.legend()
plt.show()

# Przewidywanie przyszłych wartości
fut_inp = Xtest[-1:]
tmp_inp = list(fut_inp[0,:,0])

# Inicjalizacja listy do przechowywania przewidywań
list_output = []
n_steps = 100
i = 0

# Rozpoczęcie pętli do przewidywania przyszłych wartości
while(i < 30):
    if len(tmp_inp) > n_steps:
        fut_inp = np.array(tmp_inp[-n_steps:]).reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.append(yhat[0,0])
        list_output.append(yhat[0,0])
        i += 1
    else:
        fut_inp = np.array(tmp_inp).reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.append(yhat[0,0])
        list_output.append(yhat[0,0])
        i += 1

# Przetwarzanie wyników i rysowanie wykresów
final_output = np.array(list_output).reshape(-1, 1)
final_output = norm.inverse_transform(final_output)

final_graph = np.append(ds[-len(final_output):], final_output, axis=0)
plt.figure(figsize=(12,6))
plt.plot(ds, label='Original Data')
plt.plot(np.arange(len(ds), len(ds) + len(final_output)), final_output.flatten(), label='Future Predicted Data', color='red') 
plt.legend()
plt.show()
