
# -*- coding: utf-8 -*-

from __future__ import print_function
from pydantic import IPv4AddressError
#from keras_flops import get_flops
from tensorflow import keras
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import xlwt
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from tensorflow import keras 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, concatenate, Activation
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y
  

nb_epochs = 200
batch_size = 10
NUM_CELLS =82

Acc_1=np.zeros(6)
Acc_cell = np.zeros(10)
ii=5
for ii in range(1,6):
    flist  = ['PD300']
    for each in flist:
        fname = each
    ##################### main start
        x_train, y_train = readucr('d:/07Lab_CV/03运动功能检测/帕金森3/PD300/'+fname+'_TRAIN5'+str(ii) )#+'_ad')
        #print(type(x_train))
        #print(x_train.mean())
        x_test, y_test =   readucr('d:/07Lab_CV/03运动功能检测/帕金森3//PD300/'+fname+'_TEST5'+str(ii))#+'_ad')
        nb_classes = len(np.unique(y_test))
        print(nb_classes)
        MAX_SEQUENCE_LENGTH = 319

        Y_train = keras.utils.to_categorical(y_train, nb_classes)
        Y_test = keras.utils.to_categorical(y_test, nb_classes)

        x_train_mean = x_train.mean()
        x_train_std = x_train.std()
        
        x_train = (x_train - x_train_mean)/(x_train_std)
        x_test = (x_test - x_train_mean)/(x_train_std)

        #1D时间序列
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.reshape(x_test.shape + (1,))

        print(x_train.shape)
        #ip = keras.layers.Input(x_train.shape[1:])
        
        input = Input(shape=(MAX_SEQUENCE_LENGTH,1))
        x = LSTM(NUM_CELLS)(input)
        x = Dropout(0.8)(x)
        
        out = keras.layers.Dense(nb_classes, activation='softmax')(x)

        model = keras.models.Model(inputs=input, outputs=out)
          
        optimizer = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['acc'])
          
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                          patience=10, min_lr=0.0001) 
        hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                  verbose=1, 
                  validation_data=(x_test, Y_test),
                  callbacks = [reduce_lr])
        #Print the testing results which has the lowest training loss.
        
        log = pd.DataFrame(hist.history)
        #print(model.summary())
        #flops = get_flops(model, batch_size=16)
        #print(f"FLOPS: {flops / 10 ** 9:.03} G")


        # 创建一个workbook 设置编码
        workbook = xlwt.Workbook(encoding = 'utf-8')
        # 创建一个worksheet
        worksheet = workbook.add_sheet('My Worksheet')
        #print(Y_test.shape[0])
        num = Y_test.shape[0]
        y_pred = model.predict(x_test)

        auc_score = roc_auc_score(Y_test,y_pred)
        #print("#######################################")
        print('nb_epochs = '+str(nb_epochs))
        print('batch_size = '+str(batch_size))
        #print(auc_score)
        #print("#######################################")
        for i in range(len(y_pred)):
          max_value=max(y_pred[i])
          for j in range(len(y_pred[i])):
            if max_value==y_pred[i][j]:
              y_pred[i][j]=1
            else:
              y_pred[i][j]=0
        #print(classification_report(Y_test, y_pred))

        print('lk')
        print("#######################################")
        #print(log.loc[log['loss'].idxmin]['loss'])
        print("ACC:",log.loc[log['loss'].idxmin]['val_acc'])
        print("#######################################")
        Acc_1[ii]=log.loc[log['loss'].idxmin]['val_acc']
        Acc_cell[NUM_CELLS]=log.loc[log['loss'].idxmin]['val_acc']
    ##################### main end

print(Acc_1)
print(Acc_cell)
