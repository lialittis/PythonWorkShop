import pandas as pd
import numpy as np
import random
import math
from parse import *
import os
import sklearn
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
import matplotlib.image as mpimg 
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection
from collections import defaultdict
import datetime
from datetime import timedelta
import argparse

"""
define the options of main function
"""
parser_arg = argparse.ArgumentParser(description='Create Grids Graph and Time Line')
# parser_arg.add_argument("-d",'--SetDistance',nargs='?',help="Set the distance boundary to ignore data manully",type=int, default=0)
parser_arg.add_argument("-c",'--MinCountOfPoints',nargs='?',help="Set the min number of points in one grid",type=int, default=3)
parser_arg.add_argument("-C",'--MinCountOfPointsForAll',nargs='?',help="Set the min number of points in one grid for AllInOne Graph",type=int, default=-1)
parser_arg.add_argument('-s','--SizeOfGrid', help="Set the size of each grid", nargs="?",default='0.5')
parser_arg.add_argument('--figure',help = 'Figure path of appartment', nargs="?",default='floorplan01.png')
parser_arg.add_argument('--path', help="Data path", nargs="?",default='./results/')
parser_arg.add_argument("--res", help="Results path for graph generated",nargs="?", default="./figures/")

arguments = parser_arg.parse_args()

"""
Configuration
"""

max_x = 12
min_x = -2.5
max_y = 10
min_y = -4.5
unit = float(arguments.SizeOfGrid)  #网格单元长宽m
column_num = int((max_x-min_x)//unit)
row_num = int((max_y-min_y)//unit)
map_img = mpimg.imread(arguments.figure)

# time epsilon to select the data point
epsilon = timedelta(minutes=10)
min_count = arguments.MinCountOfPoints

# define time
year = 2022
input_path = arguments.path
res_path = arguments.res
C = 0
"""
File Path Creation
"""
def createPath(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")
 
"""
Grid Generator
"""
def collectData(dict_df,key):
    df = dict_df[key]

    xs = []
    ys = []
    ts = []
    for i in range(len(df['hour'])):
        ts.append(df['time'][i])
        xs.append(df['x值'][i])
        ys.append(df['y值'][i])
    c={ "time":ts,"x":xs,'y':ys}
    data= pd.DataFrame(c)
    return data

def collectDataAllInOne(df):
    xs = []
    ys = []
    ts = []
    for i in range(len(df['hour'])):
        ts.append(df['time'][i])
        xs.append(df['x值'][i])
        ys.append(df['y值'][i])
    c={ "time":ts,"x":xs,'y':ys}
    data= pd.DataFrame(c)
    return data

def generalID(x,y,column_num,row_num):
    # 若在范围外的点，返回-1
    if x <= min_x or x >= max_x or y <= min_y or y >= max_y:
        return -1
    # 把x范围根据列数等分切割
    column = (max_x - min_x)/column_num
    # 把y范围根据行数数等分切割
    row = (max_y - min_y)/row_num
    # 二维矩阵坐标索引转换为一维ID，即： （列坐标区域（向下取整）+ 1） + （行坐标区域 * 列数）
    return int((x-min_x)//column)+ 1 + int((y-min_y)//row) * column_num

def createGrid(data):
    # add label id to the data
    data['label'] = data.apply(lambda x: generalID(x['x'], x['y'],column_num,row_num), axis = 1)

    # count values
    groups = data['label'].value_counts()

    # groups = data.groupby('label')
    # groups.size() # count the number of each grid
    # print(groups)
    

    # re-organize the grid
    grids = np.zeros([row_num,column_num])
    for i in groups.index:
        r = (i-1)//column_num  # e.g. (6 - 1) // 12 = 0; (13-1) // 12 = 1; (12-1) // 12 = 0
        c = i - column_num * r - 1 # e.g. 6 - 0 * 12 - 1= 6; 13 - 1 * 12 -1= 0; 12 - 0 * 12 - 1 = 11
        grids[r,c] = math.sqrt(groups[i])/math.sqrt(max(groups))  # Do a sqrt to make the heatmap clearer !!!
    # print(grids)
    return groups,grids

"""
Heatmap
"""
def createHeatmap(grids,figName):
    fig, ax = plt.subplots(figsize=(column_num*0.5,row_num*0.5))
    with sns.axes_style("white"):
        ax = sns.heatmap(grids, cmap='Reds',linewidths=.0,alpha=.7,xticklabels =False,square = True,yticklabels =False,mask=(grids==0.),center=0.5)
    ax.invert_yaxis()
    plt.imshow(map_img,zorder = 0, extent=[-0/unit,13/unit,-0/unit,14/unit])
    #plt.show()
    fig.savefig(res_path+figName+".png",dpi=300)
    plt.close()

# Slice By Time
"""
select label to different zone
"""
def select(dict_zones,label,zone_index,used_labels,c):
    if (label not in groups.index) or (label in used_labels) or groups[label] <=c or label <= 0 or label > row_num*column_num :
        return False
    dict_zones[zone_index].append(label)
    used_labels.append(label)
    select(dict_zones,label-column_num,zone_index,used_labels,c)
    select(dict_zones,label+column_num,zone_index,used_labels,c)
    if label%column_num != 0:
        select(dict_zones,label+1,zone_index,used_labels,c)
    if label%column_num != 1:
        select(dict_zones,label-1,zone_index,used_labels,c)
    return True

"""
get the dictionary of zones
"""
def getDictZones(groups,c):
    # choose zones of labels
    dict_zones = defaultdict(list)
    zone_index = 1
    used_labels = []
    for label in groups.index:
        if label in used_labels:
            continue
        else:
            select(dict_zones,label,zone_index,used_labels,c)
            zone_index += 1
    # print(dict_zones)
    return dict_zones
"""
slice by time, which should be continous according to epsilon
"""
def sliceByTime(time_list):
    max_interval = 1
    times_in_zone = []
    temp_count = 1
    start_time = time_list[0]
    end_time = time_list[0]
    for i in range(1,len(time_list)):
        t1 = time_list[i-1]
        t2 = time_list[i]
        time_delta = timedelta(hours=t2.hour,minutes=t2.minute) - timedelta(hours=t1.hour,minutes=t1.minute)
        if time_delta >= epsilon:
            max_interval = max(temp_count,max_interval)
            temp_count = 1
            end_time = t1
            times_in_zone.append([start_time,end_time])
            start_time = t2
        else:
            temp_count += 1
    max_interval = max(max_interval,temp_count)
    end_time = time_list[-1]
    times_in_zone.append([start_time,end_time])
    return max_interval,times_in_zone

"""
get the data of corresponding zone
"""
def getTimeIntervalsForZones(dict_zones,data):
    result = {}
    details = {}
    for zone in dict_zones:
        labels = dict_zones[zone]
        data_zone = data.loc[data['label'].isin(labels)]
        time_list = data_zone['time'].tolist()
        result[zone],details[zone] = sliceByTime(time_list)

    print("time interval in all zones", result)
    return result,details
"""
Heatmap by Time
"""

def createHeatmapByTime(intervals_in_zones,figName):
    # re-organize the grid
    new_grids = np.zeros([row_num,column_num])
    for zone in intervals_in_zones:
        for label in dict_zones[zone]:
            r = (label-1)//column_num  # e.g. (6 - 1) // 12 = 0; (13-1) // 12 = 1; (12-1) // 12 = 0
            c = label - column_num * r - 1 # e.g. 6 - 0 * 12 - 1= 6; 13 - 1 * 12 -1= 0; 12 - 0 * 12 - 1 = 11
            new_grids[r,c] = intervals_in_zones[zone]  # Do a sqrt to make the map clearer !!!
    #specify size of heatmap
    fig, ax = plt.subplots(figsize=(column_num*0.5,row_num*0.5))
    ax = sns.heatmap(new_grids, cmap='GnBu',linewidths=0,alpha=0.5,xticklabels =False,square = True,annot=False,yticklabels =False,mask=(new_grids==0.),center=0.5)
    ax.invert_yaxis()
    #plt.show()
    plt.imshow(map_img,zorder = 0, extent=[-0/unit,13/unit,-0/unit,14/unit])
    fig.savefig(res_path+figName+"_temporal.png",dpi=300)
    plt.close(fig)
    return new_grids

"""
show data on the heatmap by time
"""
def createHeatmapByTime2(new_grids,figName):
    fig, ax = plt.subplots(figsize=(column_num*0.5,row_num*0.5))
    ax = sns.heatmap(new_grids, cmap='GnBu',linewidths=0,alpha=0.5,xticklabels =False,square = True,annot=True,yticklabels =False,mask=(new_grids==0.),center=0.5)
    ax.invert_yaxis()
    #plt.show()
    plt.imshow(map_img,zorder = 0, extent=[-0/unit,13/unit,-0/unit,14/unit])
    fig.savefig(res_path+figName+"_temporal02.png",dpi=300)
    plt.close(fig)

"""
Time line
"""
def ceateTimeLine(figName,details):
    timeline = []
    for zone in details:
        for t in details[zone]:
            parsed = parse("{}_{}",key)
            my_date = parsed[1]
            #datetime.time to datetime.datetime
            date = str(dt.datetime.strptime(my_date, '%Y-%m-%d').date())
            # NOTE : start from 6 am to mid-night
            start = dt.time(6,0,0)
            end = dt.time(23,59,59)
            s = pd.to_datetime(date + " " + start.strftime("%H:%M:%S"))
            e = pd.to_datetime(date + " " + end.strftime("%H:%M:%S"))
            timeline.append((s, s, zone))
            timeline.append((e, e, zone))
            t0 = pd.to_datetime(date + " " + t[0].strftime("%H:%M:%S"))
            t1 = pd.to_datetime(date + " " + t[1].strftime("%H:%M:%S"))
            timeline.append((t0, t1, zone))

    colormapping = {}
    for zone in details:
        colormapping[zone] = "C"+str(zone)
    verts = []
    colors = []
    for d in timeline:
        v =  [(mdates.date2num(d[0]), d[2]-.4),
              (mdates.date2num(d[0]), d[2]+.4),
              (mdates.date2num(d[1]), d[2]+.4),
              (mdates.date2num(d[1]), d[2]-.4),
              (mdates.date2num(d[0]), d[2]-.4)]
        verts.append(v)
        colors.append(colormapping[d[2]])
    bars = PolyCollection(verts, facecolors=colors)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.add_collection(bars)
    ax.autoscale()
    loc = mdates.HourLocator(byhour=[6,8,10,12,14,16,18,20,22])
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    ax.set_yticks(list(details.keys()))
    ax.set_yticklabels(list(details.keys()),fontsize=16)
    # ax.annotate('race interrupted', (61, 25),
    #             xytext=(0.8, 0.9), textcoords='axes fraction',
    #             arrowprops=dict(facecolor='black', shrink=0.05),
    #             fontsize=16,
    #             horizontalalignment='right', verticalalignment='top')
    print(list(details.keys()))
    # plt.show()
    fig.savefig(res_path+figName+"_time.png",dpi=300)
    plt.close(fig)



if __name__ == "__main__":
    
    """
    Read Data
    """
    # read files
    files = [f for f in os.listdir(input_path) if f.endswith('_dataByMinute.xlsx')]
    # read data
    tags_order = []
    all_points_array = []
    dict_df = {}
    list_df = []
    # data collection for one
    data_collection = []
    
    if arguments.MinCountOfPointsForAll == -1:
        C = min_count * len(files)
    else:
        C = arguments.MinCountOfPointsForAll

    for f in files:
        # print(f)
        df = pd.read_excel(open(input_path+f,'rb'))
        # one copy of df
        df_copy = df.copy()

        parsed = parse("{}_{}_{}",f)
        #print(parsed)
        my_date = str(year)+'-'+parsed[1][:2]+'-'+parsed[1][2:]
        date = str(dt.datetime.strptime(my_date, '%Y-%m-%d').date())
        #print(my_date)
        points = []
        for i in range(len(df['hour'])):
            point = [df['x值'][i],df['y值'][i]]
            points.append(point)
            data_collection.append(point)

            # define new time for all in one
            # TODO : if necessary, to change all dataframe time
            new_time = pd.to_datetime(date + " " + df['time'][i].strftime("%H:%M:%S"))
            df_copy['time'] = new_time

        points_array = np.array(points)
        tags_order.append(parsed[0])
        all_points_array.append(points_array)
        dict_df[parsed[0]+"_"+my_date] = df
        #print(all_points_array)
        list_df.append(df_copy)
    # collect all data by concatinate all dataframe to one
    data_collection_array = np.array(data_collection)
    df_week = pd.concat(list_df,sort=False)
    # NOTE : necessary ! to remove the index for series in dataframe
    df_week.reset_index(drop=True, inplace=True) 
    print("read files :",files)
    print(len(all_points_array))
    
    nb_record = len(all_points_array)

    # For sub plots
    fig = plt.figure(figsize = (12,nb_record//4*3)) # NOTE: 可以修改出图尺寸
    # create scatter
    for i in range(0,nb_record):
        ax = plt.subplot(math.ceil(nb_record/4),4,i+1)
        # NOTE: 3 rows, and 8 columns
        #ax.text(0.5, 0.5, str((3,8,i)), fontsize=18, ha='center')
        #plt.title(my_date)
        ax.scatter(all_points_array[i][:,0],all_points_array[i][:,1],s=20)
    
    # NOTE : necessary !
    createPath(res_path)
    fig.savefig(res_path+"scatter.png",dpi=300)
    plt.close(fig)

    # For all in one plot
    # fig = plt.figure(figsize = (15, 15)) # NOTE: 可以修改出图尺寸
    # create scatter
    fig,ax = plt.subplots()
    ax.scatter(data_collection_array[:,0],data_collection_array[:,1],s=20)
    fig.savefig(res_path+"scatter_allInOne.png",dpi=300)
    plt.close(fig)

    for key in dict_df.keys():
        print(key)
        data = collectData(dict_df,key)
        groups,grids = createGrid(data)
        # print(grids)
        createHeatmap(grids,key)
        dict_zones = getDictZones(groups,min_count)
        intervals_in_zones,details = getTimeIntervalsForZones(dict_zones,data)
        new_grids = createHeatmapByTime(intervals_in_zones,key)
        createHeatmapByTime2(new_grids,key)
        ceateTimeLine(key,details)

    data_allInOne = collectDataAllInOne(df_week)
    groups,grids = createGrid(data_allInOne)
    createHeatmap(grids,"heatmap_allInOne")
    dict_zones = getDictZones(groups,C)
    intervals_in_zones,details = getTimeIntervalsForZones(dict_zones,data_allInOne)
    new_grids = createHeatmapByTime(intervals_in_zones,"heatmap_allInOne")
    createHeatmapByTime2(new_grids,"heatmap_allInOne")

