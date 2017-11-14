# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:26:17 2017

@author: xiao
"""


import pickle
import h5py as h5py
import numpy as np
import csv
import pandas as pd
import json 
import heapq
import pymongo
from bson import objectid
import math
file_path='D:/data/gram/data_index.dump'
#file_path1='D:/xunlei/data_picked/gram3_gain1000/call_index.json'
file_path_prob='D:/data/gram/data_set.h5'
#
with open(file_path, "rb") as f:
    data_id, data_call_list = pickle.load(f)
h5_file = h5py.File(file_path_prob,'r')
data_prob = h5_file['data_prob'][:]
h5_file.close()    
    
#    

def gram_info(data_none_ntdll_index_value,features_all,str_value):
    data_none_ntdll_index_dump=[]
    for data_call_single in data_none_ntdll_index_value[:]:
        data_call_single_gram=[]
        data_call_single_str = [ str(i) for i in data_call_single ]
        grams_string = [" ".join(data_call_single_str [i:i+3]) for i in xrange(len(data_call_single_str)-3+1)]
        for gram in grams_string:
            if gram in features_all:
                data_call_single_gram.append(gram)
        data_call_single_gram_list=[]      
        for data_gram in data_call_single_gram:
           data_gram_list= data_gram.rstrip().split(" ")
           data_call_single_gram_list.append(data_gram_list)
        data_single_list=[] 
        if len(data_call_single_gram_list):
            for i in data_call_single_gram_list[0]:
                data_single_list.append(i)
            for data_single in data_call_single_gram_list:
                #4 gram
                
                if data_single[0]==data_single_list[-3] and data_single[1]==data_single_list[-2] and data_single[2]==data_single_list[-1] :
                   pass   
                elif data_single[0]==data_single_list[-2] and data_single[1]==data_single_list[-1] :
                    data_single_list.append(data_single[2]) 
                    
                elif data_single[0]==data_single_list[-1] :  
                    data_single_list.append(data_single[1]) 
                    data_single_list.append(data_single[2])
                    
                else:
                    for i in data_single:
                        data_single_list.append(i) 
            data_single_list = [ int(i) for i in data_single_list]
        #change
        else:
            data_single_list=[ int(i) for i in data_call_single[:3] ]
            #pass
        
        data_single_list=np.array(data_single_list, dtype=np.int32)           
        data_none_ntdll_index_dump.append(data_single_list)
        
        
    pickle.dump(data_none_ntdll_index_dump, open('dump_%s'%str_value,'wb'))
    return data_none_ntdll_index_dump
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

def write_data(file_name,f_process_row,process_row_y):
    with open(file_name, "wb") as f:
        raw_x = pickle.dump((f_process_row,process_row_y),f)
        return raw_x

print 'start0'   
feature= pickle.load(open('feature_all','rb'))                      
#data_original_value = pickle.load(open('data_original_value','rb'))
data_original_index_dump=gram_info(data_call_list,feature,'data_original_value_test')
raw_x,raw_y=raw_process(data_prob,data_original_index_dump)
data_truncate=truncate(raw_x)
label=data_label(raw_y)
write=write_data('data_process_data_original_value_test',data_truncate,label) 
