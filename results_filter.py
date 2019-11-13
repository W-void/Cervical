import os
import json
import progressbar # pip install progressbar2

IN_JSON_PATH = '/media/watson/Documents/tianchi/cervical/cervical_inferences'
OUT_JSON_PATH = '/media/watson/Documents/tianchi/cervical/results_filter0.7' 

if not os.path.exists(OUT_JSON_PATH):
    os.makedirs(OUT_JSON_PATH)

# 置信度阈值
P_THRESHOLD = 0.7

in_json_list = os.listdir(IN_JSON_PATH)
in_json_list.sort(key=lambda x: int(x[6:-5]))

file_cnt = 0
file_total = len(in_json_list)
# 遍历json文件
for in_json_file in in_json_list:
    file_cnt += 1
    # 打开json文件载入数据list[dict]
    with open(os.path.join(IN_JSON_PATH, in_json_file), 'r') as f:
        in_json_list = json.load(f)

    out_json_list = []
    dict_cnt = 0
    dict_total = len(in_json_list)
    barPrefix = '('+str(file_cnt)+'/'+str(file_total)+')...' + in_json_file
    bar = progressbar.ProgressBar(prefix=barPrefix, max_value=dict_total).start()  
    # 遍历json数据list
    for in_json_dict in in_json_list:
        dict_cnt += 1
        bar.update(dict_cnt)
        # 过滤结果
        if in_json_dict["p"] >= P_THRESHOLD:
            out_json_list.append(in_json_dict)
    # 写入json
    out_json_filePath = os.path.join(OUT_JSON_PATH, in_json_file)
    with open(out_json_filePath, 'w') as outfile:  
        outfile.write(json.dumps(out_json_list))
    bar.finish()
