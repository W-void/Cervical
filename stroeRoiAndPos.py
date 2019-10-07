import kfbReader
import cv2
import os
import json

outpath = './RoiAndPos/'
if not output:
    os.makedirs(outpath)
    
for i in range(1):##########  10
    kfb_path = './pos_'+str(i)
    name_list = os.listdir(kfb_path)
    for filename in name_list:
        name = os.path.splitext(filename)[0]
        kfb_file = os.path.join(kfb_path,filename)
        scale = 20
        read = kfbReader.reader()
        kfbReader.reader.ReadInfo(read, kfb_file, scale, True)

        with open('./labels/'+name+'.json', 'r') as f:
            data = json.load(f)
            for j in range(len(data)):
                d = data[j]
                roi = read.ReadRoi(d["x"], d["y"], d["w"], d["h"], scale)
                cv2.imwrite(outpath+name+'-'+d['class']+'_'+ str(d["x"])+'_'+str(d["y"])+'.jpg',roi)
            
            