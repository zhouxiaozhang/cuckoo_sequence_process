# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:28:33 2017

@author: xiao
"""
import numpy as np
import pickle
import re
from bson import objectid
import h5py as h5py
import pymongo

#file_name='F:\cuckoo\data_origin.obj'
file_name_w='/home/zx/pattern/data_origin_pattern.obj'
file_path='/home/zx/open_data/data_index.dump'
file_path_prob='/home/zx/open_data/data_set.h5'
h5_file = h5py.File(file_path_prob,'r')
data_prob = h5_file['data_prob'][:]
def load_data(file_name):
    with open(file_name, "rb") as f:
        raw_x,raw_y = pickle.load(f)
        return raw_x,raw_y
    
def write_data(file_name,f_process_row,process_row_y):
    with open(file_name, "wb") as f:
        raw_x = pickle.dump((f_process_row,process_row_y),f,-1)
        return raw_x
"""\1第一个括号捕获的123123123123123越多越好"""
REPEATER = re.compile(r"((\d+,)+)\1+")
def repeated(s):
    match = re.search(REPEATER,s)
    if match:
	"""获得一个或多个分组截获的字符串；返回(start(group), end(group))"""
       return match.group() ,match.span()
    else:
       return None,None
"""重复模式保留一个123123123123123，到中止分细点"""   
REPEATER_again= re.compile(r"((\d+,)+?)\1+$")
def repeated_again(s):
    match = REPEATER_again.match(s)
    return match.group(1) if match else None
     
raw_y,raw_x=load_data(file_path)  
raw_process=[]
t=0
for sStr in raw_x[:]:
    sStr2 = ",".join(map(str, sStr)) 
    sStr2=sStr2+","
    flag=1
    sStr1=""
    while(flag==1):
	"""一次循环重复模式保留一个"""
        sStr2_len=len(sStr2) 
        group_sub,span_sub= repeated(sStr2)
        if(group_sub==None):
            sStr1=sStr1+sStr2
            flag=0
        else: 
            sub_again=repeated_again(group_sub)
            j=len(sub_again)  
            k=span_sub[0]+j
            if (sub_again==sStr2[span_sub[1]:span_sub[1]+j]):
                sStr1=sStr1+sStr2[:k]
                sStr2= sStr2[span_sub[1]+j:]
            else:
                sStr1=sStr1+sStr2[:k]
                sStr2=  sStr2[span_sub[1]:]
    list_raw = sStr1.split(',')
    del list_raw[-1]
    array_row= np.array(list_raw,dtype=np.int32)
    raw_process.append(array_row)
    t=t+1
    print t
          
write=write_data(file_name_w,raw_process,raw_y)          
            
            
            
            
             
    
    
    
    
  


     
        
        


    
    



 




