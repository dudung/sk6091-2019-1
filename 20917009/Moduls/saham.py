###########################################################################################
# Title 	: Prediksi Indeks Harga Saham dengan menggunakan Recurrent Neural Network
# Author	: Achmad M. Gani
# Aims  	: 1. Menerapkan metode neural network berbasis time series dengan program Python
#	          2. Meramal harga indeks saham di masa depan
# Input		: Matriks dengan dimensi m x n dan merupakan fungsi time series
# Output	: Grafik dan variabel dengan tipe data list berisi nilai prediksi terhadap waktu
# Outline Code	: 1. Header
#		  2. Definisi Fungsi
#		  3. Algorithma
#		  4. Prediksi / Forecasting
############################################################################################
#-------------------HEADER-----------------
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#------------------ PREDEFINE THE FUNCTIONS ---------------

def build_timeseries(mat, y_col_index):
    TIME_STEPS = 3
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
    print("length of time-series", x.shape, y.shape)
    return x, y

def trim_dataset(mat, batch_size):
    no_of_rows = mat.shape[0]% batch_size
    if(no_of_rows > 0) :
        return mat[:-no_of_rows]
    else:
        return mat
#--------------------------------------- ALGORITHMS ------------------------
#load the data and prepare the constants
BATCH_SIZE = 10
PATH = '../Dataset/BBCAMax.csv'
stock_time_series = pd.read_csv(PATH)

#------ Preprocessing -----
# drop null values in "Close"
stock_time_series = stock_time_series.dropna(subset=['Close'])

train_cols = ["Open", "High", "Low", "Close", "Volume"]
df_train, df_test = train_test_split(stock_time_series, train_size=0.8, test_size=0.2, shuffle=False)
print("Train and Test size", len(df_train), len(df_test))

#normalization
x = df_train.loc[:, train_cols].values
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x)
x_test = scaler.transform(df_test.loc[:, train_cols])

#turn into supervised learning
x_t, y_t = build_timeseries(x_train, 3)
x_t, y_t = trim_dataset(x_t, BATCH_SIZE), trim_dataset(y_t, BATCH_SIZE)
x_temp, y_temp = build_timeseries(x_test, 3)
x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE), 2)
y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE), 2)

#Build the model

lstm_model = Sequential()
lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, 3, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful= True,
                    kernel_initializer='random_uniform'))
lstm_model.add(Dropout(0.02))
#lstm_model.add(Dense(20, activation='relu'))
lstm_model.add(Dense(1, activation='linear'))
optimizer = optimizers.RMSprop(lr=0.00008)
lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)

history = lstm_model.fit(x_t, y_t, epochs=40, verbose=2, batch_size=BATCH_SIZE,
                    shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                    trim_dataset(y_val, BATCH_SIZE)))

# predict the test data
y_pred = lstm_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
y_pred = y_pred.flatten()
y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
error = mean_squared_error(y_test_t, y_pred)
print("Error is", error, y_pred.shape, y_test_t.shape)
print(y_pred[0:15])
print(y_test_t[0:15])

# convert the predicted value to range of real data
y_pred_org = (y_pred * scaler.data_range_[3]) + scaler.data_min_[3]
# min_max_scaler.inverse_transform(y_pred)
y_test_t_org = (y_test_t * scaler.data_range_[3]) + scaler.data_min_[3]
# min_max_scaler.inverse_transform(y_test_t)
print(y_pred_org[0:15])
print(y_test_t_org[0:15])

# Visualize the prediction
plt.figure()
plt.plot(y_pred_org)
plt.plot(y_test_t_org)
plt.title('Prediksi vs Harga Asli')
plt.ylabel('Harga')
plt.xlabel('Hari')
plt.legend(['Prediksi', 'Asli'], loc='upper left')
plt.show()

#Visualize the val_loss and loss
"""
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training & Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
plt.show()
"""
#forecasting attempt
"""
Check weight, 
weights, biases = model.layers[0].get_weights()
"""
