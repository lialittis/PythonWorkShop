程序使用手册

一，数据预处理

根据每时刻的位置记录excel生成按分钟或按几分钟统计的位置记录

程序 ： cleaner.py

运行方法 ：

a. 显示程序参数 ： python cleaner.py -h

usage: cleaner.py [-h] [-n [NUMBEROFTAGS]] [-i [INTERVALOFMINUTES]] [--data [DATA]] [--r [R]]

Data Cleaner

optional arguments:
  -h, --help            show this help message and exit
  -n [NUMBEROFTAGS], --NumberOfTags [NUMBEROFTAGS]
                        Set the number of tags in each file
  -i [INTERVALOFMINUTES], --IntervalOfMinutes [INTERVALOFMINUTES]
                        Set the interval of minutes to seperate
  --data [DATA]         Data path
  --r [R]               Result path

b. 示例 ：

1. python cleaner.py
（默认tags数 ：2， 默认间隔时间: 10分钟, 默认数据和存储文件夹分别是./data/ 和 ./results/）

2. python cleaner.py -n 3 -i 5 --data ./DataPath/ --r ./ResPath/

二，生成图像

程序： GridGraphGenerator.py

可以对给定文件夹内的所有byMinute的数据点进行 “依次” 以及 “联合” 成图

运行方法 ：

a. 显示程序参数 ：python GridGraphGenerator.py -h

usage: GridGraphGenerator.py [-h] [-c [MINCOUNTOFPOINTS]] [-s [SIZEOFGRID]] [--figure [FIGURE]]
                             [--path [PATH]] [--res [RES]]

Create Grids Graph and Time Line

optional arguments:
  -h, --help            show this help message and exit
  -c [MINCOUNTOFPOINTS], --MinCountOfPoints [MINCOUNTOFPOINTS]
                        Set the min number of points in one grid
  -C [MINCOUNTOFPOINTSFORALL], --MinCountOfPointsForAll [MINCOUNTOFPOINTSFORALL]
                        Set the min number of points in one grid for AllInOne Graph
  -s [SIZEOFGRID], --SizeOfGrid [SIZEOFGRID]
                        Set the size of each grid
  --figure [FIGURE]     Figure path of appartment
  --path [PATH]         Data path
  --res [RES]           Results path for graph generated

b. 示例 ：

1. python GridGraphGenerator.py
(默认每天每人的网格密度选择下限：3， 文件夹内所有天数和所有出现的人的数据总和的网格密度选择下限：3*文件数量，
默认网格大小: 0.5米，默认房间底图路径 ：floorplan01.png，默认读取byMinute数据和存储图片的文件夹分别是results和figures）

2. python GridGraphGenerator.py -c 4 -C 40 -s 1 --figure newfigure.png --path ./DataPath/ --res ./ResPath/

