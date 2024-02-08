import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

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

# Podział danych na treningowe, walidacyjne i testowe
train_size = int(len(ds_scaled) * 0.70)
val_size = int(len(ds_scaled) * 0.15)
test_size = len(ds_scaled) - train_size - val_size
ds_train, ds_val, ds_test = ds_scaled[0:train_size, :], ds_scaled[train_size:train_size + val_size, :], ds_scaled[train_size + val_size:len(ds_scaled), :]

# Przygotowanie danych treningowych, walidacyjnych i testowych
time_stamp = 100
Xtrain, Ytrain = build_ds(ds_train, time_stamp)
Xval, Yval = build_ds(ds_val, time_stamp)
Xtest, Ytest = build_ds(ds_test, time_stamp)

# Reshape danych
Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], 1)
Xval = Xval.reshape(Xval.shape[0], Xval.shape[1], 1)
Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], 1)

# Budowanie modelu
model = Sequential()
model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(time_stamp, 1)))
model.add(SimpleRNN(units=50, return_sequences=True))
model.add(SimpleRNN(units=50))
model.add(Dense(units=1, activation='linear'))

# Kompilacja i trenowanie modelu
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(Xtrain, Ytrain, validation_data=(Xval, Yval), epochs=100, batch_size=64)

# Predykcja na danych walidacyjnych i testowych
val_predict = model.predict(Xval)
test_predict = model.predict(Xtest)

# Rescaling predykcji do oryginalnych wartości
val_predict = norm.inverse_transform(val_predict)
test_predict = norm.inverse_transform(test_predict)

# Rysowanie wykresów
plt.figure(figsize=(12,6))
plt.plot(ds, label='Original Data')

# Rysowanie predykcji walidacyjnej
start_point_val = len(ds_train) - len(val_predict)
plt.plot(np.arange(start_point_val, start_point_val + len(val_predict)), val_predict.flatten(), label='Validation Predicted Data', color='green') 

# Rysowanie predykcji testowej
start_point_test = len(ds_train) + len(ds_val) - len(test_predict)
plt.plot(np.arange(start_point_test, start_point_test + len(test_predict)), test_predict.flatten(), label='Test Predicted Data', color='orange') 

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
        # Dodanie tylko pierwszej przewidzianej wartości do tmp_inp
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
