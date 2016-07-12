__author__ = 'Kirill Rudakov'
import os
import pickle
import cardio_info
import numpy as np

start_data_list = os.listdir('data/clear')

try:
    with open('pickle/analyzed_data_list.pickle', 'rb') as file:
        analyzed_data_list = pickle.load(file)
except:
    analyzed_data_list = []

HRV_stats = open('data/hrv.txt','a')

for training in start_data_list:
    if training not in analyzed_data_list:
        rr,time_axis = cardio_info.get_time_and_RR('data/clear/'+training)
        TP,LF,HF,LF_HF = cardio_info.get_frequences(rr,time_axis)
        HRV_stats.write(str(TP)+','+str(LF)+','+str(HF)+','+str(LF_HF)+'\n')
        analyzed_data_list.append(training)

with open('pickle/analyzed_data_list.pickle', 'wb') as file:
    pickle.dump(analyzed_data_list, file)

try:
    HRV_stats = open('data/hrv.txt','r')
    print(np.genfromtxt(HRV_stats, delimiter=','))[:,3]
except:
    print('Error! Empty input file!')