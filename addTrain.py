#%%
import os
import random


#%%
outpath = './PyTorch-YOLOv3-master/data/custom/'
tr = 0.8
imgPath = outpath + 'images/'
imgs = os.listdir(imgPath)
random.shuffle(imgs)
size = int(tr*len(imgs))

with open(outpath + 'train.txt', 'w') as f:
    for img in imgs[:size]:
        f.write('data/custom/images/' + img + '\n')
with open(outpath + 'valid.txt', 'w') as f:
    for img in imgs[size:]:
        f.write('data/custom/images/' + img + '\n')