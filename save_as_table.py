lang='eng'
path = "similarity/avgs/"+lang+"/"
path_to_save="similarity/avgs/"
import os
from utils import *
import numpy as np

files = os.listdir(path)
main = {}

events = open_json(path+files[0])
events = events.keys()
avgs = []
header = ''
for file in files:
    avgs.append(open_json(path+file))
    header = header + file+','
main['header'] = header
for event in events:
    temp=""
    for avg in avgs:
        temp = temp + str(np.round(avg[event]*100,decimals=1)) + ","
    main[event] = temp
save_file(path_to_save + lang+"_main_avgs_file", main)

print("done")

