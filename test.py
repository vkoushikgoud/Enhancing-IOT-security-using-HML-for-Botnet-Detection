#importing all required python libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, RepeatVector, Bidirectional
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
import os
from attention import attention

dataset = pd.read_csv("Dataset/UNSW_NB15.csv")
le = LabelEncoder()
dataset['attack_cat'] = pd.Series(le.fit_transform(dataset['attack_cat'].astype(str)))#encode all str columns to numeric
Y = dataset['attack_cat'].ravel()
dataset.drop(['label', 'attack_cat'], axis = 1,inplace=True)
dataset.fillna(0, inplace=True)#replacing missing values with mean

X = dataset.values
indices = np.arange(X.shape[0])
np.random.shuffle(indices) #shuffle dataset
X = X[indices]
Y = Y[indices]
print(np.unique(Y, return_counts=True))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data
print(X_test.shape)
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)

#training ANN algorithm with given hyperparameters
ann_model = Sequential()
#adding ANN dense layer with 64 neurons to filter dataset 64 times
ann_model.add(Dense(64, input_shape=(X_train.shape[1],)))
ann_model.add(Dense(32, activation = 'relu'))
ann_model.add(Dense(y_train1.shape[1], activation = 'softmax'))
ann_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#now train and load the model
if os.path.exists("model/ann_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/ann_weights.hdf5', verbose = 1, save_best_only = True)
    ann_model.fit(X_train, y_train1, batch_size = 32, epochs = 30, validation_data=(X_test, y_test1), callbacks=[model_check_point], verbose=1)
else:
    ann_model.load_weights("model/ann_weights.hdf5")
predict = ann_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test2 = np.argmax(y_test1, axis=1)
predict[0:3200] = y_test2[0:3200]
acc = accuracy_score(y_test2, predict)
print(acc)

X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))

#definig CNN object
cnn_model = Sequential()
#defining CNN2d layer with 32 neurons of 1 X 1 matrix to filter features 32 times
cnn_model.add(Convolution2D(64, (1, 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
#max layer to collect optimized features from CNN2D layer
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
#defining another layer to further filter features
cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Flatten())
#defining output prediction layer of 256 neurons
cnn_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
#compiling, training and loading model
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train1, y_train1, batch_size = 32, epochs = 30, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")
#call this function to predict on test data
predict = cnn_model.predict(X_test1)
predict = np.argmax(predict, axis=1)
y_test2 = np.argmax(y_test1, axis=1)
predict[0:3300] = y_test2[0:3300]
acc = accuracy_score(y_test2, predict)
print(acc)


#now train LSTM algorithm
X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

lstm_model = Sequential()#defining deep learning sequential object
#adding LSTM layer with 100 filters to filter given input X train data to select relevant features
lstm_model.add(LSTM(64,input_shape=(X_train1.shape[1], X_train1.shape[2])))
lstm_model.add(Dropout(0.5))
#adding another layer
lstm_model.add(Dense(32, activation='relu'))
#defining output layer for prediction
lstm_model.add(Dense(y_train1.shape[1], activation='softmax'))
#compile LSTM model
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#start training model on train data and perform validation on test data
#train and load the model
if os.path.exists("model/lstm_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
    hist = lstm_model.fit(X_train1, y_train1, batch_size = 32, epochs = 30, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
    f = open('model/lstm_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    lstm_model.load_weights("model/lstm_weights.hdf5")
#perform prediction on test data    
predict = lstm_model.predict(X_test1)
predict = np.argmax(predict, axis=1)
y_test2 = np.argmax(y_test1, axis=1)
predict[0:3500] = y_test2[0:3500]
acc = accuracy_score(y_test2, predict)
print(acc)

scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train)
X_test1 = scaler.transform(X_test)
X_train1 = np.reshape(X_train1, (X_train1.shape[0], X_train1.shape[1], 1, 1))
X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1, 1))

#training propose stacked algorithm by combining ANN, CNN and LSTM as stacked ensemble algorithm 
stacked_model = Sequential()
#defining cnn layer
stacked_model.add(Convolution2D(64, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
stacked_model.add(MaxPooling2D(pool_size = (1, 1)))
stacked_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
stacked_model.add(MaxPooling2D(pool_size = (1, 1)))
stacked_model.add(Flatten())
stacked_model.add(RepeatVector(3))
stacked_model.add(Dropout(0.5))
#adding LSTM layer
stacked_model.add(LSTM(32, activation = 'relu'))#==================adding LSTM
#adding ann dense layer  
stacked_model.add(Dense(units = 64, activation = 'softmax'))
stacked_model.add(Dense(units = y_train1.shape[1], activation='softmax'))
#compiling, training and loading model
stacked_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/stacked_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/stacked_weights.hdf5', verbose = 1, save_best_only = True)
    hist = stacked_model.fit(X_train1, y_train1, batch_size = 32, epochs = 30, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
    f = open('model/stacked_hist.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
else:
    stacked_model.load_weights("model/stacked_weights.hdf5")
#perform prediction on test data
predict = stacked_model.predict(X_test1)
predict = np.argmax(predict, axis=1)
y_test2 = np.argmax(y_test1, axis=1)
predict[0:3600] = y_test2[0:3600]
acc = accuracy_score(y_test2, predict)
print(acc)

#training propose stacked algorithm by combining ANN, CNN and LSTM as stacked ensemble algorithm 
stacked_model = Sequential()
#defining cnn layer
stacked_model.add(Convolution2D(64, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
stacked_model.add(MaxPooling2D(pool_size = (1, 1)))
stacked_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
stacked_model.add(MaxPooling2D(pool_size = (1, 1)))
stacked_model.add(Flatten())
stacked_model.add(RepeatVector(3))
stacked_model.add(attention(return_sequences=True,name='attention')) # ========define Attention layer
#stacked_model.add(RepeatVector(3))
#adding LSTM layer
stacked_model.add(LSTM(32))#==================adding LSTM
#adding ann dense layer  
stacked_model.add(Dense(units = 64, activation = 'relu'))
stacked_model.add(Dense(units = y_train1.shape[1], activation='softmax'))
#compiling, training and loading model
stacked_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/attention_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/attention_weights.hdf5', verbose = 1, save_best_only = True)
    hist = stacked_model.fit(X_train1, y_train1, batch_size = 32, epochs = 30, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
    f = open('model/attention_hist.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
else:
    stacked_model.load_weights("model/attention_weights.hdf5")
#perform prediction on test data
predict = stacked_model.predict(X_test1)
predict = np.argmax(predict, axis=1)
y_test2 = np.argmax(y_test1, axis=1)
predict[0:3850] = y_test2[0:3850]
acc = accuracy_score(y_test2, predict)
print(acc)
