import numpy as np
import os
import json
import kfbReader
import cv2 as cv
import pandas as pd


# %%
outpath = './RoiWithRect/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

for i in range(1):  # 10
    kfb_path = './pos_'+str(i)
    nameList = os.listdir(kfb_path)
    for filename in nameList:
        name = os.path.splitext(filename)[0]
        kfb_file = os.path.join(kfb_path, filename)
        scale = 20
        read = kfbReader.reader()
        read.ReadInfo(kfb_file, scale, True)

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
            roi = read.ReadRoi(x0, y0, w0, h0, scale)
            for j, pos in dfPos.iterrows():
                x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
                cv.rectangle(roi, (x-x0, y-y0),
                             (x-x0+w, y-y0+h), (0, 0, 255), 10)

            cv.imwrite(outpath+name+'-'+item['class']+'_' +
                       str(item["x"])+'_'+str(item["y"])+'.jpg', roi)
