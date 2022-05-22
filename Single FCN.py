
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:11:19 2016

@author: stephen
"""
 
from __future__ import print_function
#from keras_flops import get_flops
from tensorflow import keras
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import xlwt
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y
  
nb_epochs = 100
batch_size = 10

Acc_1=np.zeros(6)
for ii in range(5,6):
    print(ii);
    flist  = ['PD300']
    for each in flist:
        fname = each
    ##################### main start
        x_train, y_train = readucr('d:/07Lab_CV/03运动功能检测/帕金森3/PD300/'+fname+'_TRAIN5'+str(ii) +'_ad')
        x_test, y_test =   readucr('d:/07Lab_CV/03运动功能检测/帕金森3//PD300/'+fname+'_TEST5'+str(ii)+'_ad')
        nb_classes = len(np.unique(y_test))
        print(nb_classes)
         
    
        Y_train = keras.utils.to_categorical(y_train, nb_classes)
        Y_test = keras.utils.to_categorical(y_test, nb_classes)

        x_train_mean = x_train.mean()#all data--each line
        x_train_std = x_train.std()
        x_train = (x_train - x_train_mean)/(x_train_std)
        
        x_test = (x_test - x_train_mean)/(x_train_std)

        #1D time series
        x_train = x_train.reshape(x_train.shape + (1,1,))
        x_test = x_test.reshape(x_test.shape + (1,1,))

        x = keras.layers.Input(x_train.shape[1:])
        
        
        #drop_out = Dropout(0.2)(x)
        conv1 = keras.layers.Conv2D(64, 8, 1, padding='same')(x)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)

        #    drop_out = Dropout(0.2)(conv1)
        conv2 = keras.layers.Conv2D(128, 5, 1, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        #    drop_out = Dropout(0.2)(conv2)
        conv3 = keras.layers.Conv2D(64, 3, 1, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)
        
      
        full = keras.layers.GlobalAveragePooling2D()(conv3)
        out = keras.layers.Dense(nb_classes, activation='softmax')(full)


        model = keras.models.Model(inputs=x, outputs=out)
          
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
        print(model.summary())
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
        
        print('nb_epochs = '+str(nb_epochs))
        print('batch_size = '+str(batch_size))
        print(auc_score)
        #print("#######################################")
        for i in range(len(y_pred)):
          max_value=max(y_pred[i])
          for j in range(len(y_pred[i])):
            if max_value==y_pred[i][j]:
              y_pred[i][j]=1
            else:
              y_pred[i][j]=0
        print(classification_report(Y_test, y_pred))


        for i  in range(0,1):
            y_row=Y_test[i]
            p_row=y_pred[i]
            worksheet.write(i,0,str(np.argmax(y_row)))
            worksheet.write(i,1,str(np.argmax(p_row)))
        # 保存
        workbook.save('test1.xls')

        
        print("ACC:",log.loc[log['loss'].idxmin]['val_acc'])
        
        Acc_1[ii]=log.loc[log['loss'].idxmin]['val_acc']
    ##################### main end
  
print(Acc_1)
