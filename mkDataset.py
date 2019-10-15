import kfbReader
import cv2 as cv
import os
import json
import numpy as np 
import pandas as pd 


outpath = './data/VOCdevkit2007/VOC2007/'
imgPath = outpath + 'JPEGImages/'
labelsPath = outpath + 'labels/'

if not os.path.exists(imgPath):
    os.makedirs(imgPath)
if not os.path.exists(labelsPath):
    os.makedirs(labelsPath)

imgSize = 256
num = 0

for i in range(1):  # 10
    kfbPath = './pos_'+str(i)
    nameList = os.listdir(kfbPath)
    for filename in nameList:
        name = os.path.splitext(filename)[0]
        kfb_file = os.path.join(kfbPath, filename)
        scale = 20
        read = kfbReader.reader()
        kfbReader.reader.ReadInfo(read, kfb_file, scale, True)

        with open('./labels/'+name+'.json', 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(np.zeros((len(data), 5)),
                          columns=['x', 'y', 'w', 'h', 'class'], dtype=object)
        for i, item in enumerate(data):
            df.loc[i] = item

        dfRoi = df.loc[df['class'] == 'roi']
        dfPos = df.loc[df['class'] == 'pos']

        for i, item in dfRoi.iterrows():
            x0, y0, w0, h0 = item['x'], item['y'], item['w'], item['h']
            for j, pos in dfPos.iterrows():
                x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
                if((x > x0) and (y > y0) and (x+w < x0+w0) and (y+h < y0+h0)):
                    size = imgSize
                    while((w > size) or (h>size)):
                        size *= 2
                    dx, dy = np.random.randint(0, size-w), np.random.randint(0, size-h)
                    x_, y_ = x - dx, y - dy
                    png = read.ReadRoi(x_, y_, size, size, scale)
                    cv.imwrite(imgPath+"%07d"%num+'.png', png)

                    # xmin, ymin, xmax, ymax = dx, dy, w+dx, h+dy
                    with open(labelsPath+"%07d"%num+'.txt', 'w') as f:
                        # f.write("%d %d %d %d %d\n"%(xmin, ymin, xmax, ymax, 1))
                        # 检查有没有包含其他的pos
                        for k, p in dfPos.iterrows():
                            x, y, w, h = p['x'], p['y'], p['w'], p['h']
                            xmin, ymin, xmax, ymax = np.clip([x-x_, y-y_, x+w-x_, y+h-y_], 0, size-1)
                            if((xmax-xmin>10) and (ymax-ymin>10)):
                                f.write("%d %d %d %d %d\n"%(xmin, ymin, xmax, ymax, 1))
                    num += 1
                