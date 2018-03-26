# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import h5py as h5py
import numpy as np
import csv
import pandas as pd
import json 
import pymongo
import heapq
from bson import objectid

file_path='D:/data/gram/data_index.dump'
file_path1='D:/data/gram/call_index.json'
file_path_prob='D:/data/gram/data_set.h5'

with open(file_path, "rb") as f:
    data_id, data_call_list = pickle.load(f)
h5_file = h5py.File(file_path_prob,'r')
data_prob = h5_file['data_prob'][:]
h5_file.close()    
    
    
data_index={}
for json_data in open(file_path1):
    data = json.loads(json_data)
    data_index[data['index']]=data['name']

data_call_list_reverse_list=[]   
for sample in data_call_list:
    data_call_list_reverse=[]
    for id in sample:
        data_call_list_reverse.append(data_index[id])   
    data_call_list_reverse_list.append(data_call_list_reverse)
    
# function index          
data_reverse={}
for key,value in data_index.iteritems():
    data_reverse[value]=key

data_original=[]                
#data_original
data_original_value=[]
for sample_value in data_call_list_reverse_list:
    data_original_index_temp_value=[]
    for id in sample_value:
        if id in data_reverse:
           data_original_index_temp_value.append(data_reverse[id])   
    data_original_value.append(data_original_index_temp_value) 
pickle.dump(data_original_value, open('data_original_value','wb'))


    
def load_label(data_prob ,label):
    result = []
    for row in range(len(data_prob)):
        if int(round(data_prob[row]))== label:
            result.append(row)
    return result

# generate grams dictionary for one file
#data_call_single = data_call_list[0]
#n-gram 字典
def grams_dict(index_single, N,data_index_function): 
    data_call_single = data_index_function[index_single]
    grams_string = [" ".join(data_call_single [i:i+N]) for i in range(len(data_call_single )-N+1)]
    tree = dict()
    for gram in grams_string:
        if gram not in tree:
            tree[gram] = 1 
    return tree

def grams_dict_value(index_single, N,data_none_index_value): 
    data_call_single = data_none_index_value[index_single]
    data_call_single_str = [ str(i) for i in data_call_single ]
    grams_string = [" ".join(data_call_single_str [i:i+N]) for i in range(len(data_call_single_str)-N+1)]
    tree = dict()
    for gram in grams_string:
        if gram not in tree:
            tree[gram] = 1 
    return tree



 #add up ngram dictionaries
def reduce_dict_value(f_labels,data_none_index_value):
    result = dict()
    for f_name in f_labels:
        d = grams_dict_value(f_name,3,data_none_index_value)
        for k,v in d.iteritems():
            if k in result:
                result[k] += v 
            else:
                result[k] = v
        del d

    return result
 #add up ngram dictionaries
def reduce_dict(f_labels,data_index_function):
    result = dict()
    for f_name in f_labels:
        d = grams_dict(f_name, 3,data_index_function)
        for k,v in d.iteritems():
            if k in result:
                result[k] += v 
            else:
                result[k] = v
        del d

    return result
#   
# heap to get the top 100,000 features.
def Heap_top(dictionary, label,str_value, num = 10000):
    heap = [(0,'tmp')]* num # initialize the heap
    root = heap[0]
    for ngram,count in dictionary.iteritems():
            if count > root[0]:
                root = heapq.heapreplace(heap, (count, ngram))
    pickle.dump(heap, open('ngram_%i_top%i_%s'%(label,num,str_value),'wb'))
    return heap

label = int(0)
print ("Gathering 3 grams, Class 0 out of 0...")
f_labels = load_label(data_prob , label)
heap0_original=Heap_top(reduce_dict_value(f_labels,data_original_value),label,'data_original_value')

label = int(1)
print ("Gathering 3 grams, Class %i out of 1..."%label)
f_labels = load_label(data_prob , label)
heap1_original=Heap_top(reduce_dict_value(f_labels,data_original_value),label,'data_original_value')



     
    


    
    
    
