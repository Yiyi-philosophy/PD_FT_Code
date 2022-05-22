#从txt读入numpy 行
from asyncio.windows_events import NULL
import numpy as np
import pandas as pd

import os



data1 = pd.read_excel('D:\\07Lab_CV\\03运动功能检测\\帕金森3\\result_data2.xlsx',1)
data1 = np.array(data1,dtype=str)
print(data1.shape)

data = np.zeros((360,30*50+1))
data[0:359,0:1]=data1

data_n = np.zeros((360,30*50+1))



#i=1
for i in range(1,90):
    #file = open('D:\\07Lab_CV\\03运动功能检测\\帕金森3\\result_all_1\\'+ str(i) +'-r_W2xEX_VFI.txt', mode='r')
    reg0 = np.loadtxt('D:\\07Lab_CV\\03运动功能检测\\帕金森3\\result_all_2\\'+ str(i) +'-l_W2xEX_VFI.txt',dtype=np.float32)
    data[4*i][1:reg0.size+1] = reg0  #left hand
    reg1 = np.loadtxt('D:\\07Lab_CV\\03运动功能检测\\帕金森3\\result_all_2\\'+ str(i) +'-l1_W2xEX_VFI.txt',dtype=np.float32)
    data[4*i+1][1:reg1.size+1] = reg1  #left hand 1

    reg3 = np.loadtxt('D:\\07Lab_CV\\03运动功能检测\\帕金森3\\result_all_2\\'+ str(i) +'-r_W2xEX_VFI.txt',dtype=np.float32)
    data[4*i+2][1:reg3.size+1] = reg3  #right hand
    reg4 = np.loadtxt('D:\\07Lab_CV\\03运动功能检测\\帕金森3\\result_all_2\\'+ str(i) +'-r1_W2xEX_VFI.txt',dtype=np.float32)
    data[4*i+3][1:reg4.size+1] = reg4  #right hand 1

    print(str(i)+"ok")

print(data)



#归一化，截断
def normalization(datingDatamat):
    max_arr = datingDatamat.max(axis=0)
    min_arr = datingDatamat.min(axis=0)
    ranges = max_arr - min_arr
    norDataSet = np.zeros(datingDatamat.shape)
    m = datingDatamat.shape[0]
    norDataSet = datingDatamat - np.tile(min_arr, (m, 1))
    norDataSet = norDataSet/np.tile(ranges,(m,1))
    return norDataSet



data_n[0:359,0] = data[0:359,0]
data_n[0:359,1:20*50] = normalization(data[0:359,1:20*50])
print(data_n)
print(data_n.shape)

print(data_n.shape)
#写入excel
np.savetxt("D:\\07Lab_CV\\03运动功能检测\\帕金森3\\test_f90.csv", data_n[0:359,0:300], delimiter=',')

