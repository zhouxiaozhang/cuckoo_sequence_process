# -*- coding: utf-8 -*-
"""
Created on Sat Jun 03 22:58:46 2017

@author: xiao
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:19:07 2017

@author: xiao
"""


import h5py as h5py
import numpy as np
import csv
import pandas as pd
import json 
import heapq
import pickle
import math
from bson import objectid
import pymongo


def load_data(file_name):
    with open(file_name, "rb") as f:
        raw_x,raw_y = pickle.load(f)
        return raw_x,raw_y
file_path='D:/data/gram/data_index.dump'
file_path_prob='D:/data/gram/data_set.h5'
h5_file = h5py.File(file_path_prob,'r')
data_prob = h5_file['data_prob'][:]
h5_file.close() 
with open(file_path, "rb") as f:
    data_id, data_call_list = pickle.load(f)
data_original_value_gram = pickle.load(open('dump_data_original_value','rb'))

def raw_process(data_str,data_prob,data_value):
#array float
    raw_x=[]
    raw_y=[]
    for data,tag in zip(data_value,data_prob):
        if len(data):
            raw_x.append(data)
            tag=np.array(tag, dtype=np.float32)
            raw_y.append(tag)
    pickle.dump((raw_x,raw_y), open('data_process1_%s'%data_str,'wb'))
    return raw_x,raw_y
 
raw_x_4,raw_y_4=raw_process('data_original_value_gram',data_prob,data_original_value_gram) 



