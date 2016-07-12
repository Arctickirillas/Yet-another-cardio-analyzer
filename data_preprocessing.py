__author__ = 'Kirill Rudakov'
import os
import numpy as np
import pickle

start_data_list = os.listdir('data/dirty')

try:
    with open('pickle/data_list.pickle', 'rb') as file:
        data_list = pickle.load(file)
except:
    data_list = []

for training in start_data_list:
    if training not in data_list:
        dirty_path, clear_path = 'data/dirty/'+training, 'data/clear/'+training
        dirty, clear = open(dirty_path,'rb'), open(clear_path,'wb')

        for line in dirty:
            line = [float(i) for i in line.split(',')]
            if len(line)==3:
                clear.write(str(line[1])+','+str(line[2])+'\n')
        clear.close(),dirty.close()
        os.remove(dirty_path)
        data_list.append(training)

with open('pickle/data_list.pickle', 'wb') as file:
    pickle.dump(data_list, file)

