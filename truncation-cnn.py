# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:07:24 2017

@author: xiao
"""

import numpy as np
import pickle
import h5py as h5py
import numpy as np
import csv
import pandas as pd
import json 
import heapq
import pickle
import math
from bson import objectid
from sklearn.model_selection import train_test_split 

import h5py as h5py
file_path='D:/xunlei/data_picked/gram3_gain1000/data_index.dump'
file_path1='D:/xunlei/data_picked/gram3_gain1000/call_index.json'
file_path_api='D:/xunlei/data_picked/gram3_gain1000/win_api.csv'
file_path_prob='D:/xunlei/data_picked/gram3_gain1000/data_set.h5'
#file_path1='/home/zx/gram3_gain1000/data_process1_data_none_ntdll_index_value_gram'
#file_path2='/home/zx/gram3_gain1000/data_process1_data_none_ntdll_merge_index_value_gram'
#file_path3='/home/zx/gram3_gain1000/data_process1_data_ntdll_index_value_gram'
file_path4='D:/xunlei/data_picked/gram3_gain1000/data_process_data_original_value'
#file_path5='/home/zx/gram3_gain1000/data_process1_data_merge'
#file_path6='/home/zx/gram3_gain1000/data_process1_data_call_list_gram'
##
with open(file_path, "rb") as f:
    data_id, data_call_list = pickle.load(f)
h5_file = h5py.File(file_path_prob,'r')
data_prob = h5_file['data_prob'][:]
h5_file.close() 

def raw_process(data_prob,data_value):
#array float
    raw_x=[]
    raw_y=[]
    for data,tag in zip(data_value,data_prob):
        if len(data):
            raw_x.append(data)
            tag=np.array(tag, dtype=np.float32)
            raw_y.append(tag)
    return raw_x,raw_y


def load_data(file_name):
    with open(file_name, "rb") as f:
        raw_x,raw_y = pickle.load(f)
        return raw_x,raw_y
    
def write_data(file_name,f_process_row,process_row_y):
    with open(file_name, "wb") as f:
        raw_x = pickle.dump((f_process_row,process_row_y),f)
        return raw_x

   
def truncate (data):
    data_truncate=[] 
    for data_cut in data:
        if len(data_cut) > 1000:
            data_cut = data_cut[:1000]
        else:
            data_cut=np.lib.pad(data_cut,(0,1000-len(data_cut)),"constant",constant_values=(0,0))
        data_truncate.append(data_cut)
    return data_truncate
def data_label(labels):
    tmp=[]
    for label in labels:
        if label>0.5:
            label=np.array([0,1])
        else:
            label=np.array([1,0])
        tmp.append(label)
    return tmp

   
##
##raw_x,raw_y=raw_process(data_prob,data_call_list)
#raw_x1,raw_y1=load_data(file_path1)
##raw_x,raw_y=load_data(file_name_w) 
#data_truncate1=truncate(raw_x1)
#label1=data_label(raw_y1)
#
#write=write_data('data_process_data_none_ntdll_index_value',data_truncate1,label1) 
#
#raw_x2,raw_y2=load_data(file_path2)
##raw_x,raw_y=load_data(file_name_w) 
#data_truncate2=truncate(raw_x2)
#label2=data_label(raw_y2)
#write=write_data('data_process_data_none_ntdll_merge_index_value',data_truncate2,label2) 
#
#raw_x3,raw_y3=load_data(file_path3)
##raw_x,raw_y=load_data(file_name_w) 
#data_truncate3=truncate(raw_x3)
#label3=data_label(raw_y3)
#write=write_data('data_process_data_ntdll_index_value',data_truncate3,label3) 
#raw_x_3,raw_y_3=load_data('data_process_data_ntdll_index_value')
#
def batch_iter(x_data,y_data, batch_size, num_epochs, shuffle=True):
    print "a"
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    
#    y_data.shape=(1,len(y_data))
#    y_data=np.transpose(y_data)
    data_size =len(x_data)
    num_batches_per_epoch = int((len(x_data)-1)/batch_size)+1
    print "a"
    print num_batches_per_epoch 
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices=np.random.permutation(data_size)
            shuffled_data_x=x_data[shuffle_indices]
            shuffled_data_y=y_data[shuffle_indices]
        else:
            shuffled_data_x=x_data
            shuffled_data_y=y_data
        for batch_num in range(num_batches_per_epoch):
            start_index=batch_num *batch_size
            #print batch_size
            #print start_index
            end_index=min((batch_num+1)*batch_size,data_size)
           # print shuffled_data_x[start_index:end_index]
           # print end_index
            yield shuffled_data_x[start_index:end_index],shuffled_data_y[start_index:end_index]
raw_x4,raw_y4=load_data(file_path4)
raw_x4=truncate(raw_x4)
write=write_data('data_process_data_original_value',raw_x4,raw_y4) 
raw_x5,raw_y5=load_data('data_process_data_original_value')
#l=len(raw_x5)
#train_x=[]
#train_y=[]
#test_x=[]
#test_y=[]
#pos_1=int(l*0.9)
#train_x=raw_x5[:pos_1]
#train_y=raw_y5[:pos_1]
#test_x=raw_x5[pos_1:]
#test_y=raw_y5[pos_1:]
##x_data=np.array(train_x)
##y_data=np.array(train_y)
##
##b=len(y_data) 
##y_data.shape=(1,len(x_data))
##y_data=np.transpose(y_data)
#
#batces1=batch_iter(train_x,train_y,128,1)
#print "b"
##raw_x,raw_y=load_data(file_name_w) 
#for batcx,batcy in batces1:
#    print batcx
#    print batcy
#    a=0
data_truncate4=truncate(raw_x4)
label4=data_label(raw_y4)
x1_data=np.array(data_truncate4)
y1_data=np.array(label4)
l=len(data_truncate4)
train_x=[]
train_y=[]
test_x=[]
test_y=[]
pos_1=int(l*0.9)
train_x=data_truncate4[:pos_1]
train_y=label4[:pos_1]
test_x=data_truncate4[pos_1:]
test_y=label4[pos_1:]

write=write_data('data_process_data_original_value_train',train_x,train_y) 
write=write_data('data_process_data_original_value_test',test_x,test_y) 
x,y=load_data('data_process_data_original_value_train')
x_train,x_dev,y_train,y_dev=train_test_split(x,y,test_size=0.1)


batcx,batcy=batch_iter(train_x,train_y,len(train_x),1)
##num_batches_per_epoch = int(len(x_data)-1/batch_size)
#
##batces_x= batch_iter(x_train,y_train,64,100)
#for batcx,batcy in batces:
#    print batcx
#    print batcy
#    a=0
    

#raw_x6,raw_y6=load_data(file_path6)
##raw_x,raw_y=load_data(file_name_w) 
#data_truncate6=truncate(raw_x6)
#label6=data_label(raw_y6)
#write=write_data('data_process_data_original_value_',data_truncate6,label6) 

##
#raw_x5,raw_y5=load_data(file_path5)
#
##raw_x5,raw_y5=load_data('data_merge')
##raw_x,raw_y=load_data(file_name_w) 
#data_truncate5=truncate(raw_x5)
#label5=data_label(raw_y5)
#write=write_data('data_process_data_merge_1',data_truncate5,label5) 
#
#
#raw_x6,raw_y6=load_data(file_path6)
#
##raw_x5,raw_y5=load_data('data_merge')
##raw_x,raw_y=load_data(file_name_w) 
#data_truncate6=truncate(raw_x6)
#label6=data_label(raw_y6)
#write=write_data('data_process_data_call_list_value',data_truncate6,label6)
#
##
#raw_x7,raw_y7=load_data('data_process_data_merge_1')
#raw_x8,raw_y8=load_data('data_process_data_call_list_value')

#raw_x,raw_y=load_data('data_merge_1')
#raw_t,raw_k=load_data('data_orign')
#data_tr=data_truncate[3712:4224]
#
#data_t=raw_k[3712:4224]
#train_data = []
#test_data = []
#valid_data = []
#for i in range(len(data_truncate)):
#    if(i%9==0):
#        test_data.append(data_truncate[i])  
#    elif(i%10==0):
#        valid_data.append(data_truncate[i])
#    else:
#       train_data.append(data_truncate[i]) 
#    
#raw_x, raw_y= load_data(file_name_c)



