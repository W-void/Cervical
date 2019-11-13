#%%
import kfbReader
import cv2 as cv


#%%
path = './pos_0/T2019_53.kfb'
scale = 20
read = kfbReader.reader()
read.ReadInfo(path, scale, True)
print(read.getWidth(), read.getHeight())

#%%
roi = read.ReadRoi(48000, 46000, 512, 512, scale)
cv.imshow('roi', roi)
cv.waitKey(0)
cv.destroyAllWindows()

#%%
height = read.getHeight()
width = read.getWidth()
scale = read.getReadScale()
print('height:{0}, width:{1}, scale:{2}'.format(
    height, width, scale))

#%%
