import os


dirs = os.listdir('.')
for d in dirs:
    if os.path.isdir(d) and d[0] not in ['.', '_']:
        f = open('./'+d+'/.gitkeep', 'w')
        f.close()