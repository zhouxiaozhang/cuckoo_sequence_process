# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:12:31 2017

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

file_path_prob='D:/data/gram/data_set.h5'
h5_file = h5py.File(file_path_prob,'r')
data_prob = h5_file['data_prob'][:]
h5_file.close()  
def join_ngrams(str,num = 10000):
    dict_all = dict()
    for c in range(0,2):
        #print "merging %i out of 2"%c
        heap = pickle.load(open('ngram_%i_top%i_%s'%(c,num,str),'rb'))
        while heap:
            count, gram = heapq.heappop(heap)
            if gram not in dict_all:
                dict_all[gram] = [0]*2
            dict_all[gram][c] = count
    return dict_all


# load data
def num_instances(data_prob ,label):
    p = 0
    n = 0
    for row in range(len(data_prob)):
        if int(round(data_prob[row]))== label:
            p += 1
        else:
            n += 1
    return p,n
def entropy(p,n):
    p_ratio = float(p)/(p+n)
    n_ratio = float(n)/(p+n)
    return -p_ratio*math.log(p_ratio) - n_ratio * math.log(n_ratio)

def info_gain(p0,n0,p1,n1,p,n):
    return entropy(p,n) - float(p0+n0)/(p+n)*entropy(p0,n0) - float(p1+n1)/(p+n)*entropy(p1,n1)


def Heap_gain(p, n, class_label, dict_all, num_features = 1000, gain_minimum_bar = -100000):
    heap = [(gain_minimum_bar, 'gain_bar')] * num_features
    root = heap[0]
    for gram, count_list in dict_all.iteritems():
        p1 = count_list[class_label]
        n1 = count_list[abs(class_label-1)]#sum(count_list[:(class_label)] + count_list[class_label:])
        p0,n0 = p - p1, n - n1
        if p1*p0*n1*n0 != 0:
            gain = info_gain(p0,n0,p1,n1,p,n)
            if gain > root[0]:
                root = heapq.heapreplace(heap, (gain, gram))
    #return heap
    return [i[1] for i in heap]

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
        else:
            pass
        
        data_single_list=np.array(data_single_list, dtype=np.int32)           
        data_none_ntdll_index_dump.append(data_single_list)
        
        
    pickle.dump(data_none_ntdll_index_dump, open('dump_%s'%str_value,'wb'))


print 'start0' 
dict_all_4 = join_ngrams('data_original_value')
features_all_4 = []
p, n = num_instances(data_prob ,0)
features_all_4  += Heap_gain(p,n,0,dict_all_4) # 750 * 9 
pickle.dump(features_all_4, open('feature_all','wb'))                         
data_original_value = pickle.load(open('data_original_value','rb'))
gram_info(data_original_value,features_all_4,'data_original_value')
       
    
    
    
       
    
        
       
        
        
        
        
        
            
    
    
    
    
    
            
            
            





                          


                          

