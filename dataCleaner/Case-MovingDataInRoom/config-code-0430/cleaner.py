import pandas as pd
import numpy as np
import datetime
import math
import os
import matplotlib.pyplot as plt
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook
import sys
import argparse


"""
define the options of main function
"""
parser_arg = argparse.ArgumentParser(description='Data Cleaner')
# parser_arg.add_argument("-d",'--SetDistance',nargs='?',help="Set the distance boundary to ignore data manully",type=int, default=0)
# parser_arg.add_argument("-nl",'--NumberToLookBack',nargs='?',help="",type=int, default=0)
parser_arg.add_argument('-n','--NumberOfTags', help="Set the number of tags in each file", nargs="?",type=int,default=2)
parser_arg.add_argument('-I','--IntervalOfMinutes', help="Set the interval of minutes to seperate", nargs="?",type=int,default=1)
parser_arg.add_argument('-i','--IntervalOfSeconds', help="Set the interval of seconds to seperate", nargs="?",type=int,default=1)
parser_arg.add_argument('--data', help="Data path", nargs="?",default='./data/')
parser_arg.add_argument("--r", help="Result path",nargs="?", default="./results/")

arguments = parser_arg.parse_args()


files = [f for f in os.listdir(arguments.data)]
list_data = []
for f in files:
    path = arguments.data+f
    for i in range(arguments.NumberOfTags):
        tag_name = 'tag'+str(i)
        store_file_name = tag_name+'_'+f
        print(store_file_name)
        list_data.append((pd.read_excel(path,tag_name,index_col="时间"),store_file_name))

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def calcul_distance(x1,y1,x2,y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2)

def is_close(x1,y1,x2,y2,limit):
    if calcul_distance(x1,y1,x2,y2) < limit:
        return True
    return False

def clean_zeros(df):
    print("with zeros in seconds :",df.shape)
    df.plot()
    df_non_zeros = df[(df['x值'] != 0.0) & (df['y值'] !=0.0)]
    print("with zeros elimated in seconds :", df_non_zeros.shape)
    df_non_zeros.plot()
    return df_non_zeros

"""
Clustering function [not finished]
"""
def clustering(df,n_lookback):
    minutes = df['minute'] # read minutes column
    hours = df['hour']
    xs = df['x值']
    ys = df['y值']
    df_clustered = df.copy()
    for i in range(n_lookback,len(minutes)):
        if is_close(xs[i],ys[i],xs[i-n_lookback],ys[i-n_lookback]):
            df_clustered['x值'][i] = df_clustered['x值'][i-n_lookback]
            df_clustered['y值'][i] = df_clustered['y值'][i-n_lookback]
    return df_clustered

def extend_data_sec(df,IntervalOfSeconds):
    modulo = int(60/IntervalOfSeconds)
    seconds = df['second']
    minutes = df['minute'] # read minutes column
    hours = df['hour']
    xs = df['x值']
    ys = df['y值']
    df_extended = df.copy()
    for i in range(1,len(seconds)):
        current_hour = hours[i-1]
        current_minute = minutes[i-1]

        dif_hour = hours[i] - current_hour # minute difference
        dif_minute = minutes[i] - current_minute # minute difference
        nb_to_add = (seconds[i] - seconds[i-1] - 1)
        nb_to_add_min = dif_minute -1
        nb_to_add += dif_minute*modulo + dif_hour*modulo*60
        max_nb = nb_to_add
        while(nb_to_add != 0):
            dif = 1/(nb_to_add+1)
            index = i+dif-1
            current_second = (seconds[i-1]+max_nb-nb_to_add+1)%modulo
            if current_second == 0:
                current_minute += 1
            if current_minute == 60:
                current_minute = 0
                current_hour += 1
            if current_minute >=60:
                print(current_hour,current_minute,current_second)
            add_line = pd.DataFrame({'hour':current_hour, 'minute':current_minute,'second':current_second, 'x值':xs[i-1], 'y值':ys[i-1]},index=[index])
            df_extended = df_extended.append(add_line,ignore_index=False)
            nb_to_add -= 1
    df_extended = df_extended.sort_index().reset_index(drop=True)
    # print(df_extended)
    print("after extended in seconds:",df_extended.shape)
    return df_extended


def extend_data(df,IntervalOfMinutes):
    name_minute = 'minute' # if IntervalOfMinutes == 1 else str(IntervalOfMinutes)+'minutes'
    modulo = int(60/IntervalOfMinutes)
    minutes = df[name_minute] # read minutes column
    hours = df['hour']
    xs = df['x值']
    ys = df['y值']
    df_extended = df.copy()
    for i in range(1,len(minutes)):
        current_hour = hours[i-1]
        dif_hour = hours[i] - current_hour # hour difference
        nb_to_add = (minutes[i] - minutes[i-1] - 1)
        if dif_hour >= 1:
            nb_to_add = nb_to_add + dif_hour*modulo
        if nb_to_add < 0:
            if minutes[i] == modulo-1:
                continue # go to the next hour
            else: # this hour is not finished
                nb_to_add = modulo-1 - minutes[i-1]
                max_nb = nb_to_add
        else:
            max_nb = nb_to_add
        while(nb_to_add != 0):
            dif = 1/(nb_to_add+1)
            index = i+dif-1
            current_minute = (minutes[i-1]+max_nb-nb_to_add+1)%modulo
            if current_minute == 0:
                current_hour += 1
            add_line = pd.DataFrame({'hour':current_hour, name_minute:current_minute, 'x值':xs[i-1], 'y值':ys[i-1]},index=[index])
            df_extended = df_extended.append(add_line,ignore_index=False)
            nb_to_add -= 1
    df_extended = df_extended.sort_index().reset_index(drop=True)
    # print(df_extended)
    print("after extended in minutes:",df_extended.shape)
    return df_extended

def dataBySecond(df):
    df_n = df.groupby(df.index).mean() # mean value in each second
    # df_n = df.groupby(df.index).max() # max value in each second
    # df_n = df.groupby(df.index).first() # first value in each second
    df_n.plot()
    return df_n

def dataBySecondAndStore(df_seconds,path,IntervalOfSeconds,store_dict):
    df = df_seconds.copy()
    df['time'] = df.index
    hours = []
    mins = []
    seconds = []
    times = []
    for time in df.index:
        hours.append(time.hour)
        mins.append(time.minute)
        seconds.append(math.floor(time.second/IntervalOfSeconds))
    df['hour'] = hours
    df['minute'] = mins
    df['second'] = seconds
 
    df_n = df.groupby(['hour','minute','second']).mean() # mean value in each second
    df_n.reset_index(inplace=True)

    df_n = extend_data_sec(df_n,IntervalOfSeconds)
    for i in range(len(df_n['minute'])):
        times.append(datetime.time(df_n['hour'][i],df_n['minute'][i],df_n['second'][i]*IntervalOfSeconds))
    df_n.insert(3,"time",times,True)
    #
    # try to store by workbook
    wb = Workbook()
    ws = wb.active
    for r in dataframe_to_rows(df_n, index=False, header=True):
        ws.append(r)
    # ws.delete_rows(ws.min_row+1, 1)
    store_suffix = '_dataBySecond.xlsx' if IntervalOfSeconds == 1 else '_dataBy'+str(IntervalOfSeconds)+'Seconds.xlsx'
    wb.save(store_dict +  remove_suffix(path,'.xlsx')+store_suffix)
    #df_n.to_excel('results/'+  remove_suffix(path,'.xlsx')+'_dataByMinute.xlsx')
    return df_n

def dataByMinute(df_seconds,path,IntervalOfMinutes,store_dict):
    name_minute = 'minute' # if IntervalOfMinutes == 1 else str(IntervalOfMinutes)+'minutes'

    # index to column
    df = df_seconds.copy()
    df['time'] = df.index
    hours = []
    mins = []
    times = []

    for time in df.index:
        hours.append(time.hour)
        mins.append(math.floor(time.minute/IntervalOfMinutes))
    df['hour'] = hours
    df[name_minute] = mins
    # print(df)
    df_n = df.groupby(['hour',name_minute]).mean() # mean value in each minute
    # df_n = df.groupby(['hour','minute']).first() # first value in eahc minute
    # df_n = df.groupby(['hour','minute']).max() # max value in each minute
    # print(df_n)
    # df_n.plot().figure.savefig(store_dict + remove_suffix(path,'.xlsx')+'position_groupby_'+str(IntervalOfMinutes)+'_min(s).png')
    df_n.reset_index(inplace=True)

    # To extend the dataframe
    df_n = extend_data(df_n,IntervalOfMinutes)
    for i in range(len(df_n[name_minute])):
        times.append(datetime.time(df_n['hour'][i],df_n[name_minute][i]*IntervalOfMinutes))
    df_n.insert(2,"time",times,True)
    # try to store by workbook
    wb = Workbook()
    ws = wb.active
    for r in dataframe_to_rows(df_n, index=False, header=True):
        ws.append(r)
    # ws.delete_rows(ws.min_row+1, 1)
    store_suffix = '_dataByMinute.xlsx' if IntervalOfMinutes == 1 else '_dataBy'+str(IntervalOfMinutes)+'Minutes.xlsx'
    wb.save(store_dict +  remove_suffix(path,'.xlsx')+store_suffix)
    #df_n.to_excel('results/'+  remove_suffix(path,'.xlsx')+'_dataByMinute.xlsx')
    return df_n


store_dict = arguments.r

for df,path in list_data:
    df_new = clean_zeros(df)
    df_seconds = dataBySecond(df_new)
    df_seconds_stored = dataBySecondAndStore(df_seconds,path,arguments.IntervalOfSeconds,store_dict)
    df_minutes = dataByMinute(df_seconds,path,arguments.IntervalOfMinutes,store_dict)
    # df_10minutes = dataByMinute(df_seconds,path,arguments.IntervalOfMinutes,store_dict)

