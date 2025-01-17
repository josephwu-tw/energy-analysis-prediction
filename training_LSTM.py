import keras
from keras.models import Sequential # type: ignore
from keras.layers import Dense # type: ignore
from keras.layers import LSTM # type: ignore
from keras.layers import Dropout # type: ignore
import data_processing as dpss
import matplotlib.pyplot as plt

# get X
X_train = dpss.X_train
X_test = dpss.X_test

# build model
keras.backend.clear_session()
lstm_model = Sequential()

lstm_model.add(keras.Input(shape = (X_train.shape[1], X_train.shape[2])))
lstm_model.add(LSTM(units = 32, activation = 'relu', return_sequences = True))
lstm_model.add(Dropout(0.1))

lstm_model.add(LSTM(units = 16, activation = 'relu', return_sequences = True))
lstm_model.add(Dropout(0.1))

lstm_model.add(LSTM(units = 16, activation = 'relu'))
lstm_model.add(Dropout(0.1))

lstm_model.add(Dense(units = 1))

lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

lstm_model.summary()

# training
history = lstm_model.fit(X_train, dpss.y_train,
                         validation_split = 0.2,
                         verbose = 1,
                         epochs = 15,
                         batch_size = 128,
                         shuffle = True)

# plot train history
plt.figure(figsize = (8,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train History - LSTM')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.show()

# prediction performance
y_pred = lstm_model.predict(X_test)
y_pred = dpss.y_scaler.inverse_transform(y_pred.reshape(-1,1))

dpss.pred_real_plot(y_pred, model = 'LSTM')

# save model
lstm_model.save("./models/lstm_model.keras")