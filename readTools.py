#%%
import kfbReader
import cv2 as cv


#%%
path = 'D:\Data\gongjingai\pos_0\pos_0\T2019_53.kfb'
scale = 20
read = kfbReader.reader()
read.ReadInfo(path, scale, True)

#%%
roi = read.ReadRoi(10240, 10240, 512, 512, scale)
cv.imshow('roi', roi)
cv.waitKey(0)

#%%
height = read.getHeight()
width = read.getWidth()
scale = read.getReadScale()
print('height:{0}, width:{1}, scale:{2}'.format(
    height, width, scale))

#%%
