#%%
import numpy as np
import os
import json
import kfbReader
import cv2 as cv
import pandas as pd 


#%%
labelPath = './labels/'
kbfPath = './pos_0/'
files = os.listdir(kbfPath)
with open(labelPath+files[0][:-3]+'json', 'r') as f:
    data = json.load(f)

#%%
read = kfbReader.reader()
scale = 20
read.ReadInfo(kbfPath+files[0], scale, True)
df = pd.DataFrame(np.zeros((len(data), 5)), columns=['x', 'y', 'w', 'h', 'class'], dtype=object)
# df.loc[0] = data[0] # 改变数据类型
for i, item in enumerate(data):
    df.loc[i] = item

dfRoi = df.loc[df['class'] == 'roi']
dfPos = df.loc[df['class'] == 'pos']

for i, item in dfRoi.iterrows():
    x0, y0, w0, h0 = item['x'], item['y'], item['w'], item['h']
    roi = read.ReadRoi(x0, y0, w0, h0, scale)
    for j, pos in dfPos.iterrows():
        x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
        cv.rectangle(roi, (x-x0, y-y0), (x-x0+w, y-y0+h), (0,0,255), 20)
    cv.namedWindow('roi', 0)
    cv.imshow('roi', roi)
    cv.waitKey(0)

cv.destroyAllWindows()
