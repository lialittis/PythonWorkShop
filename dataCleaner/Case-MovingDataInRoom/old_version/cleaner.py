import pandas as pd
import numpy as np
import datetime
import math
import os
import matplotlib.pyplot as plt
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook

files = [f for f in os.listdir("./data")]
list_data = []
for f in files:
    path = "data/"+f
    list_data.append((pd.read_excel(path,'tag0',index_col="时间"),'tag0_'+f))
    list_data.append((pd.read_excel(path,'tag1',index_col="时间"),'tag1_'+f))

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def clean_zeros(df):
    print("with zeros in seconds :",df.shape)
    df.plot()
    df_non_zeros = df[(df['x值'] != 0.0) & (df['y值'] !=0.0)]
    print("with zeros elimated in seconds :", df_non_zeros.shape)
    df_non_zeros.plot()
    return df_non_zeros

def extend_data(df):
    minutes = df['minute'] # read minutes column
    hours = df['hour']
    xs = df['x值']
    ys = df['y值']
    df_extended = df.copy()
    print(df_extended)

    for i in range(1,len(minutes)):
        nb_to_add = (minutes[i] + 60 - minutes[i-1] - 1) % 60 # to consider the first few minutes of next hour
        if nb_to_add < 0:
            if minutes[i] == 59:
                continue # go to the next hour
            else: # this hour is not finished
                nb_to_add = 59 - minutes[i-1]
                max_nb = nb_to_add
        else:
            max_nb = nb_to_add
        while(nb_to_add != 0):
            dif = 1/(nb_to_add+1)
            index = i+dif-1
            add_line = pd.DataFrame({'hour':hours[i-1], 'minute':(minutes[i-1]+max_nb-nb_to_add+1)%60, 'x值':xs[i-1], 'y值':ys[i-1]},index=[index])
            df_extended = df_extended.append(add_line,ignore_index=False)
            nb_to_add -= 1
    print(df_extended)
    df_extended = df_extended.sort_index().reset_index(drop=True)
    print(df_extended)
    print("after extended in minutes:",df_extended.shape)

    return df_extended


def extend_data_for10(df):
    minutes = df['10minutes'] # read minutes column
    hours = df['hour']
    xs = df['x值']
    ys = df['y值']
    df_extended = df.copy()
    print(df_extended)

    for i in range(1,len(minutes)):
        nb_to_add = (minutes[i] + 6 - minutes[i-1] - 1) % 6 # to consider the first few minutes of next hour
        if nb_to_add < 0:
            if minutes[i] == 5:
                continue # go to the next hour
            else: # this hour is not finished
                nb_to_add = 5 - minutes[i-1]
                max_nb = nb_to_add
        else:
            max_nb = nb_to_add
        while(nb_to_add != 0):
            dif = 1/(nb_to_add+1)
            index = i+dif-1
            add_line = pd.DataFrame({'hour':hours[i-1], '10minutes':(minutes[i-1]+max_nb-nb_to_add+1)%6, 'x值':xs[i-1], 'y值':ys[i-1]},index=[index])
            df_extended = df_extended.append(add_line,ignore_index=False)
            nb_to_add -= 1
    print(df_extended)
    df_extended = df_extended.sort_index().reset_index(drop=True)
    print(df_extended)
    print("after extended in minutes:",df_extended.shape)

    return df_extended

def dataBySecond(df):
    df_n = df.groupby(df.index).mean() # mean value in each second
    # df_n = df.groupby(df.index).max() # max value in each second
    # df_n = df.groupby(df.index).first() # first value in each second
    df_n.plot()
    return df_n

def dataByMinute(df_seconds,path):
    # index to column
    df = df_seconds.copy()
    df['time'] = df.index
    hours = []
    mins = []
    times = []
    for time in df.index:
        hours.append(time.hour)
        mins.append(time.minute)
    df['hour'] = hours
    df['minute'] = mins
    # print(df)
    df_n = df.groupby(['hour','minute']).mean() # mean value in each minute
    # df_n = df.groupby(['hour','minute']).first() # first value in eahc minute
    # df_n = df.groupby(['hour','minute']).max() # max value in each minute
    # print(df_n)
    df_n.plot()
    df_n.reset_index(inplace=True)

    # To extend the dataframe
    df_n = extend_data(df_n)
    for i in range(len(df_n['minute'])):
        times.append(datetime.time(df_n['hour'][i],df_n['minute'][i]))
    df_n.insert(2,"time",times,True)
    # try to store by workbook
    wb = Workbook()
    ws = wb.active
    for r in dataframe_to_rows(df_n, index=True, header=True):
        ws.append(r)
    wb.save('results/'+  remove_suffix(path,'.xlsx')+'_dataByMinute.xlsx')
    #df_n.to_excel('results/'+  remove_suffix(path,'.xlsx')+'_dataByMinute.xlsx')
    return df_n

def dataBy10Minutes(df_seconds,path):
    df = df_seconds.copy()
    # index to column
    df['time'] = df.index
    hours = []
    _10mins = []
    times = []
    for time in df.index:
        hours.append(time.hour)
        _10mins.append(math.floor(time.minute/10))
    df['hour'] = hours
    df['10minutes'] = _10mins
    # print(df)
    # df_n = df.groupby(['hour','10minutes']).first() # first value in eahc minute 
    df_n = df.groupby(['hour','10minutes']).mean() # mean value in each minute
    # df_n = df.groupby(['hour','10minutes']).max() # max value in each minute
    df_n.plot().figure.savefig('results/'+  remove_suffix(path,'.xlsx')+'position_groupby_10_mins.png')
    # print(df_n)
    df_n.reset_index(inplace=True)
    # To extend the dataframe
    df_n = extend_data_for10(df_n)
    for i in range(len(df_n['10minutes'])):
        times.append(datetime.time(df_n['hour'][i],df_n['10minutes'][i]))
    df_n.insert(2,"time",times,True)
    # try to store by workbook
    wb = Workbook()
    ws = wb.active
    for r in dataframe_to_rows(df_n, index=True, header=True):
        ws.append(r)
    wb.save('results/'+  remove_suffix(path,'.xlsx')+'_dataBy10Minute.xlsx')
    #df_n.to_excel('results/'+  remove_suffix(path,'.xlsx') +'_dataBy10Minute.xlsx')
    return df_n

for df,path in list_data:
    df_new = clean_zeros(df)
    df_seconds = dataBySecond(df_new)
    df_minutes = dataByMinute(df_seconds,path)
    df_10minutes = dataBy10Minutes(df_seconds,path)

