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

import h5py as h5py
file_path='D:/data/gram/data_index.dump'
file_path1='D:/data/gram/call_index.json'
file_path_prob='D:/data/gram/data_set.h5'
file_path4='D:/data/gram/data_process1_data_original_value_gram'
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
            label=1.0
        else:
            label=0.0
        tmp.append(label)
    labels=np.array(tmp)
    labels.shape=(1,len(labels))  
    batch_label=np.transpose(labels)   
    return batch_label

#
raw_x4,raw_y4=load_data(file_path4)
data_truncate4=truncate(raw_x4)
label4=data_label(raw_y4)
write=write_data('data_process_data_original_value',data_truncate4,label4) 




