import pandas as pd
import numpy as np
from parse import *
import os
from matplotlib import pyplot as plt
#%matplotlib auto
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import timedelta
import helper
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
    for i in range(len(df['time'])):
        ts.append(df['time'][i])
        xs.append(df['x值'][i])
        ys.append(df['y值'][i])
    c={ "time":ts,"x":xs,'y':ys}
    data= pd.DataFrame(c)
    print("After Collect Data:\n",data)
    return data

##########################################
# Data pre-view
##########################################


##########################################
# Data re-clean
##########################################

"""
Shift the data/position accoding to the prevision and family configuration
"""
def shift(family,df):
    for scenario in family.scenario:
        # make sure that in each scenario
        # there is only one shift linearly
        recordsInRoom = defaultdict(list)
        if scenario.name == "Shift":
            zones = scenario.zones
            for index, row in df.iterrows():
                x = row['x值']
                y = row['y值']
                m = "DoNothing"
                f = -1
                for zone in scenario.zones:
                    if helper.inZone(x,y,zone):
                        m = zone.method
                        if m == "DoNothing":
                            break
                        # print("===============> use:"+m)
                        if m.endswith("Linearly"):
                            recordsInRoom[(m,zone)].append(index)
                        else:
                            f = helper.methods[m]
                            # deal with simple shift
                            x,y = f(x,y,zone)
                            df.at[index,'x值'] = x
                            df.at[index,'y值'] = y
                        break
            # print(recordsInRoom)
            for key in recordsInRoom.keys():
                indices = recordsInRoom[key]
                m = key[0]
                zone = key[1]
                f = helper.methods[m]
                temp_df = df.iloc[indices]
                points = []
                for i,r in temp_df.iterrows():
                    # print(i,r)
                    x = r['x值']
                    y = r['y值']
                    points.append([x,y])
                new_points = f(points,zone)
                for i in range(len(indices)):
                    index = indices[i]
                    df.at[index,'x值'] = new_points[i][0]
                    df.at[index,'y值'] = new_points[i][1]
        else:
            continue
    return df

def collectDataAllInOne(df):
    xs = []
    ys = []
    ts = []
    for i in range(len(df['time'])):
        ts.append(df['time'][i])
        xs.append(df['x值'][i])
        ys.append(df['y值'][i])
    c={ "time":ts,"x":xs,'y':ys}
    data= pd.DataFrame(c)
    print(data)
    return data

def generalID(x,y,column_num,row_num,max_x,min_x,max_y,min_y):
    # 若在范围外的点，返回-1
    if x <= min_x or x >= max_x or y <= min_y or y >= max_y:
        return -1
    # 把x范围根据列数等分切割
    column = (max_x - min_x)/column_num
    # 把y范围根据行数数等分切割
    row = (max_y - min_y)/row_num
    # 二维矩阵坐标索引转换为一维ID，即： （列坐标区域（向下取整）+ 1） + （行坐标区域 * 列数）
    return int((x-min_x)//column)+ 1 + int((y-min_y)//row) * column_num

# 去除出门不在家的点
def WashData(data):
    del_list=[]
    for idx,row in data.iterrows():
        if row['label']==21:
            del_list.append(idx)
    for i in del_list:
        data=data.drop(i)
    data.reset_index(inplace=True,drop=True)
    return data

# s=0.5m 门口网格为325
def createGrid(data,figName,column_num,row_num,max_x,min_x,max_y,min_y):
    # add label id to the data    label: 当前数据处于哪一个网格
    data['label'] = data.apply(lambda x: generalID(x['x'], x['y'],column_num,row_num,max_x,min_x,max_y,min_y), axis = 1)
    data = WashData(data)

    # count values
    groups = data['label'].value_counts() # 统计网格中的点数
    groups[21]=0   # 删去门口对应的网格中所有点数 （0.5m-325；1m-67）
    
    #print(groups)
    
    # re-organize the grid
    grids = np.zeros([row_num,column_num])
    for i in groups.index:
        r = (i-1)//column_num  # e.g. (6 - 1) // 12 = 0; (13-1) // 12 = 1; (12-1) // 12 = 0
        c = i - column_num * r - 1 # e.g. 6 - 0 * 12 - 1= 6; 13 - 1 * 12 -1= 0; 12 - 0 * 12 - 1 = 11
        # \sqrt{当前网格中的点数/最大网格中的点数}
        grids[r,c] = math.sqrt(groups[i])/math.sqrt(max(groups))  # Do a sqrt to make the heatmap clearer !!!
    
    
    #data.to_excel (res_path+figName+"_dataWithLabel.xls", index = False, header=True)
   
    return groups,grids


def createScatter(data_collection_array,figName,map_img,res_path,max_x,min_x,max_y,min_y):
    fig,ax0 = plt.subplots(figsize=(max_x-min_x,max_y-min_y))
    plt.imshow(map_img,extent=[min_x-0.1,max_x,min_y-0.1,max_y]) # 0.3 is used to balance the influence of wight space suround the image
    #ax1 = fig.add_axes()
    ax0.scatter(data_collection_array[:,0],data_collection_array[:,1],s=20)
    fig.savefig(res_path+figName+"_scatter.png",dpi=300)
    plt.close()

"""
Heatmap
"""
def createHeatmap(grids,figName,map_img,res_path,unit,max_x,min_x,max_y,min_y):
    fig, ax = plt.subplots(figsize=(max_x-min_x,max_y-min_y)) #column_num*0.5,row_num*0.5))
    with sns.axes_style("white"):
        ax = sns.heatmap(grids, cmap='Reds',linewidths=.0,alpha=.7,xticklabels =False,square = True,yticklabels =False,mask=(grids==0.),center=0.5)
    ax.invert_yaxis()
    plt.imshow(map_img,zorder = 0,  extent=[-0.1,(max_x-min_x)/unit-0.2,-0.1,(max_y-min_y)/unit-0.3] )#extent=[x_left/unit,x_right/unit,y_down/unit,y_up/unit])
    #plt.show()
    fig.savefig(res_path+figName+".png",dpi=300)
    plt.close()


# Slice By Time
"""
slice by time, which should be continous according to epsilon
"""
def sliceByTime(time_list,epsilon):
    max_interval = 1 # 该区域内的最长连续停留时间
    times_in_zone = []
    temp_count = 1
    start_time = time_list[0]
    # print("start time :",start_time)
    end_time = time_list[0]
    for i in range(1,len(time_list)):
        t1 = time_list[i-1]
        t2 = time_list[i]
        # time_list中前后两个数据的时间差
        #time_delta = timedelta(hours=t2.hour,minutes=t2.minute) - timedelta(hours=t1.hour,minutes=t1.minute)

        time_delta = t2 - t1
        
        if time_delta.total_seconds()/60 >= epsilon or t2.day != t1.day : #epsilon: # 时间差大于设定值，即在此期间离开该zone，停止计时
            print(t2,t1,time_delta)
            max_interval = max(temp_count,max_interval)
            temp_count = 1
            end_time = t1
            times_in_zone.append([start_time,end_time])
            start_time = t2 # 重设初始时间，开始下一轮计时
        else: # 时间差小于设定值，此期间内一直在该zone中，故计时+1分钟
            temp_count += 1
    max_interval = max(max_interval,temp_count)
    end_time = time_list[i-1]
    times_in_zone.append([start_time,end_time])
    return max_interval,times_in_zone

"""
get the data of corresponding zone
results: key=zone名称 value=最大连续停留时长
details: key=zone名称 value=所有停留时段 times_in_zone=[[start1,end1],[start2,end2],.....]
"""
def getTimeIntervalsForZones(dict_zones,data,epsilon):
    result = {}
    details = {}
    for zone in dict_zones:
        labels = dict_zones[zone]
        data_zone = data.loc[data['label'].isin(labels)] # 在每个zone网格中的数据
        time_list = data_zone['time'].tolist() # 时间段
        # print("time list:",time_list)
        result[zone],details[zone] = sliceByTime(time_list,epsilon)

    #print("time interval in all zones", result)
    return result,details

"""
Heatmap by Time
"""
# 无标签的区域分割
def createHeatmapByTime(dict_zones,figName,map_img,res_path,row_num,column_num,unit,max_x,min_x,max_y,min_y,groups):
    # re-organize the grid
    new_grids = np.zeros([row_num,column_num])
    zone_grids = np.zeros([row_num,column_num])
    for zone in dict_zones.keys():
        for label in dict_zones[zone]:
            r = (label-1)//column_num  # e.g. (6 - 1) // 12 = 0; (13-1) // 12 = 1; (12-1) // 12 = 0
            c = label - column_num * r - 1 # e.g. 6 - 0 * 12 - 1= 6; 13 - 1 * 12 -1= 0; 12 - 0 * 12 - 1 = 11
            new_grids[r,c] = math.sqrt(groups[label])/25 # Do a sqrt to make the map clearer !!!
            zone_grids[r,c] = zone
    #specify size of heatmap
    fig, ax = plt.subplots(figsize=(max_x-min_x,max_y-min_y))
    ax = sns.heatmap(new_grids, cmap='YlGnBu',linewidths=0,alpha=0.5,xticklabels =False,square = True,annot=zone_grids,yticklabels =False,mask=(new_grids==0.),center=0.5)
    ax.invert_yaxis()
    #plt.show()
    plt.imshow(map_img,zorder = 0, extent=[-0.1,(max_x-min_x)/unit-0.2,-0.1,(max_y-min_y)/unit-0.3] ) # extent=[x_left/unit,x_right/unit,y_down/unit,y_up/unit])
    fig.savefig(res_path+figName+"_temporal.png",dpi=300)
    plt.close(fig)
    return new_grids

"""
Don't show data on the heatmap by time
"""
# 有标签的区域分割
def createHeatmapByTime2(new_grids,figName,map_img,res_path,unit,max_x,min_x,max_y,min_y):
    fig, ax = plt.subplots(figsize=(max_x-min_x,max_y-min_y))
    ax = sns.heatmap(new_grids, cmap='YlGnBu',linewidths=0,alpha=0.5,xticklabels =False,square = True,annot=False,yticklabels =False,mask=(new_grids==0.),center=0.5)
    ax.invert_yaxis()
    #plt.show()
    plt.imshow(map_img,zorder = 0, extent= [-0.1,(max_x-min_x)/unit-0.2,-0.1,(max_y-min_y)/unit-0.3])# [x_left/unit,x_right/unit,y_down/unit,y_up/unit])
    fig.savefig(res_path+figName+"_temporal02.png",dpi=300)
    plt.close(fig)

"""
Time line
"""

def storeTimeLine(timeline,filename):
    dict_zone_timelines = defaultdict(list)
    starts = []
    ends = []
    zones = []
    periods = []
    for s,e,z in timeline:
        if s == e : continue
        starts.append(s)
        ends.append(e)
        zones.append(z)
        periods.append((e-s).total_seconds() //60)
    dict_zone_timelines['Zone'] = zones
    dict_zone_timelines['Start'] = starts
    dict_zone_timelines['End'] = ends
    dict_zone_timelines['Period'] = periods
    df = pd.DataFrame.from_dict(dict_zone_timelines)
    df.sort_values(by=['Zone'])
    print("Store Time Line:\n",df)
    helper.saveDFtoWB(df,filename)

def createTimeLine(key,details,res_path):
    timeline = []
    for zone in details:
        for t in details[zone]:
            # t0 = pd.to_datetime(t[0].strftime("%H:%M:%S"))
            # t1 = pd.to_datetime(t[1].strftime("%H:%M:%S"))
            print("Example of time:",t[0])
            timeline.append((t[0], t[1], zone))

    storeTimeLine(timeline,res_path+key+"_time.xlsx")

'''
每一天的分区表示——聚类+画成热力图
'''


'''
筛选出网格中点数达标的，去除不达标的网格中的点
input: data:原数据   groups:每个网格的点数   c:最小达标点数
output: choice_data 筛选后的数据
'''
def choose_spots(data,groups,c):
    valid_label=[]
    choice_data=data.copy()
    for i in groups.index:
        if groups[i] >= c:
            valid_label.append(i)
        
    # print(valid_label) 
    for idx,row in choice_data.iterrows():
        if row['label'] not in valid_label:
            choice_data=choice_data.drop(idx, axis=0)
    choice_data.reset_index(drop = True, inplace=True)
    return choice_data

'''
由轮廓系数得出聚类数
'''
def get_Num_from_Scores(Scores):
    idx = Scores.index(max(Scores)) # 最大轮廓系数时的索引号
    max_S = max(Scores)
    for i in range(idx,len(Scores)): # 是否有更优索引
        if (max_S-Scores[i])<0.05:
            idx = i
    print(max_S)
    print(Scores)
    return idx+2

'''
用于获得合适的聚类数
input: x,y:数据点的xy坐标  key:当天日期
output: 返回值：当天的聚类数
'''
def get_K(x,y,key,map_img,res_path,max_x,min_x,max_y,min_y):
    Scores = []  # 存放轮廓系数
    SSE = []
    X=[]
    for i in range(len(x)):
        X.append([x[i],y[i]])

    for k in range(2, 7):
        model = KMeans(n_clusters=k)  # 构造聚类器
        model.fit(X)
        y_pre = model.predict(X)
        centroid = model.cluster_centers_
        Scores.append(silhouette_score(X, y_pre))
        SSE.append(model.inertia_)

        if key=='All':
            fig,ax0 = plt.subplots(figsize=(max_x-min_x,max_y-min_y))
            plt.imshow(map_img,extent=[min_x-0.1,max_x,min_y-0.1,max_y]) 
            ax0.scatter(x,y, c=y_pre)
            ax0.scatter(centroid[:, 0], centroid[:, 1], marker='x', s=100, c='black')
            imname='k='+str(k)
            fig.savefig(res_path+key+"_"+imname+"_zone.png",dpi=300)
        
    
    X = range(2, 7)
    
    plt.figure() # 重置画布
    plt.plot(X, Scores, 'o-')
    plt.xlabel('k')
    plt.ylabel('Scores')
    plt.savefig(res_path+key+"_Scores.png",dpi=300)

    plt.figure()
    plt.plot(X, SSE, 'o-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.savefig(res_path+key+"_SSE.png",dpi=300)
    
    return get_Num_from_Scores(Scores)

'''
进行聚类
input: x,y:数据点的xy坐标  key:日期
output:
centroid: 聚类中心坐标
y_pre: 每个数据点对应的分区
'''
def K_m(x,y,key,number_cluster,map_img,res_path,max_x,min_x,max_y,min_y):
    X=[]
    for i in range(len(x)):
        X.append([x[i],y[i]])

    # 确定聚类数
    k=get_K(x,y,key, map_img,res_path,max_x,min_x,max_y,min_y)
    number_cluster[key]=k # 将聚类数存入字典

    model = KMeans(n_clusters=k)  # 构造聚类器
    model.fit(X)
    y_pre = model.predict(X)
    centroid = model.cluster_centers_ # 聚类中心

    # 画图
    fig,ax0 = plt.subplots(figsize=(max_x-min_x,max_y-min_y))
    plt.imshow(map_img,extent=[min_x-0.1,max_x,min_y-0.1,max_y]) 
    ax0.scatter(x,y, c=y_pre)
    ax0.scatter(centroid[:, 0], centroid[:, 1], marker='x', s=100, c='black')
    imname='finalK='+str(k)
    fig.savefig(res_path+key+"_"+imname+"_zone.png",dpi=300)
    return centroid,y_pre


'''
获得{value=[网格label],key=所属分区}的字典
input: choice_data内容为：x坐标 y坐标 网格label 所属分区zone
'''
def get_dict_zones(choice_data,k): #(数据，聚类数)
    dict_zones={}
    all_list=[]
    for i in range(k):
        all_list.append([])
    for idx,row in choice_data.iterrows():
        if row['label'] not in all_list[row['zone']]:
            all_list[row['zone']].append(row['label'])
    for i in range(k):
        dict_zones[i]=all_list[i]
    return dict_zones

'''
统计分区面积
'''
def get_zone_info(key,dict_zones,res_path):
    name=key
    zone_data=[]
    TotalArea=0
    for key in dict_zones.keys():
        zone_data.append([key,dict_zones[key],len(dict_zones[key])])
        TotalArea=TotalArea+len(dict_zones[key])
    #TotalArea=TotalArea*0.5*0.5 # 0.5m的网格
    info=str(TotalArea)
    zones_info=pd.DataFrame(zone_data,columns=["zone","labels","Area"])

    # 画柱状图
    plt.figure()
    p1=plt.bar(zones_info["zone"],zones_info["Area"] ,0.4,color="green")
    plt.bar_label(p1, label_type='edge')
    plt.xlabel('zone')
    plt.ylabel('Area')
    plt.savefig(res_path+name+"_Area="+info+".png",dpi=300)
    return zones_info

'''
所有分析/图片生成都在这个函数里
withTimeLine=1:同时生成timeline [只针对每天分析] key=tag1_2022-02-05 类型
'''

def FinalAnalysis(data,key,min_count,number_cluster,epsilon,row_num,column_num,map_img,res_path,unit,max_x,min_x,max_y,min_y,groups,withTimeLine=0):
    
    groups,grids = createGrid(data,key,column_num,row_num,max_x,min_x,max_y,min_y)
    data=WashData(data)
    print(data)
    createHeatmap(grids,key,map_img,res_path,unit,max_x,min_x,max_y,min_y)

    # 删去点数小于min_count的网格中的点——剩余的点更加集中，便于聚类
    choice_data=choose_spots(data,groups,min_count)
    x=choice_data['x']
    y=choice_data['y']
    
    # centers: 聚类后每一类的中心点坐标——根据中心点划定zone的范围
    centers,y_pre = K_m(x,y,key,number_cluster,map_img,res_path,max_x,min_x,max_y,min_y)
    choice_data['zone']=y_pre # 每个点的分区
    
    dict_zones= get_dict_zones(choice_data,number_cluster[key])
    zones_info=get_zone_info(key,dict_zones,res_path)
    intervals_in_zones,details = getTimeIntervalsForZones(dict_zones,data,epsilon)
    # print("intervals_in_zones:",intervals_in_zones)
    # print("details",details)
    new_grids = createHeatmapByTime(dict_zones,key,map_img,res_path,row_num,column_num,unit,max_x,min_x,max_y,min_y,groups)
    createHeatmapByTime2(new_grids,key,map_img,res_path,unit,max_x,min_x,max_y,min_y)
    
    # This part could be useless
    if withTimeLine:
        createTimeLine(key,details,res_path)
    return zones_info

#totaltimes.to_excel(res_path+'totaltimes.xlsx',index = False) 