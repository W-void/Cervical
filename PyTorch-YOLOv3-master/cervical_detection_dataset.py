# =========================================================
# @purpose: inferencing images cropped by sliding window 
#           and writing results into json file
# @date：   2019/11
# @version: v1.0
# @author： Xu Huasheng
# @github： 
# @How to run: move cervical_detection.py to $mmdetecton_root
#              cd $mmdetecton_root (~/anaconda3/envs/pytorch/mmdetection)
#              conda activate pytorch
#              python cervical_detection.py
# ==========================================================
#from mmdet.apis import init_detector, inference_detector, show_result
from models import *
from utils.utils import *
from utils.datasets import *
import os
import argparse
import progressbar
import kfbReader
import cv2
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim



def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection inference')
    parser.add_argument('--cfg', 
                        dest='config',
                        help='inference config file path',
                        type=str, 
                        default=None
                        )
    parser.add_argument('--checkpoint', 
                        dest='checkpoint',
                        help='checkpoint file path',
                        type=str, 
                        default=None
                        )
    parser.add_argument('--in', 
                        dest='img_input',
                        help='the input path of image to inference',
                        type=str, 
                        default=None
                        )
    parser.add_argument('--out', 
                        dest='img_output',
                        help='the output path of image that has inferenced',
                        type=str, 
                        default=None
                        )

    args = parser.parse_args()
    return args
 
def sliding_window(image_w, image_h, window_size, step_size):
    l = []
    for y in range(0, image_h - window_size, step_size):
        for x in range(0, image_w - window_size, step_size):
            l.append((x, y, window_size, window_size))
    return l

class ImageFolder(Dataset):
    def __init__(self, kfbread, winSize=600, imgSize=416, stepSize=500):
        self.kfbread = kfbread
        self.fullImg_width = self.kfbread.getWidth()                 # 读取全图的宽度
        self.fullImg_heigth = self.kfbread.getHeight()               # 读取全图的高度
        self.winSize = winSize
        self.imgSize = imgSize
        self.stepSize = stepSize
        self.iters = sliding_window(self.fullImg_width, self.fullImg_heigth, self.winSize, self.stepSize)

    def __getitem__(self, index):
        x, y, w, h = self.iters[index]
        window_img = self.kfbread.ReadRoi(x, y, w, h, 20)
        input_imgs = transforms.ToTensor()(window_img)
        # print(input_imgs.shape)
        # F.interpolate 默认输入的前两维是 batch 和 channel，所以不会对前两维做采样
        input_imgs = F.interpolate(input_imgs.unsqueeze(0), size=416, mode="nearest").squeeze(0)
        return x, y, input_imgs

    def __len__(self):
        return len(self.iters)



def main():
    # 默认路径
    #CONFIG_FILE = 'configs/faster_rcnn_r50_fpn_1x.py' # 模型的配置文件
    #CHECKPOINT_FILE = 'work_dirs/faster_rcnn_r50_fpn_1x/latest.pth' # 训练好的模型权重
    KFB_PATH = '../pos_0'    # kfb文件路径
    TEMP_RESULT_PATH = './temp_result'   # 临时滑窗检测结果
    OUT_JSON_PATH = './result'   # json输出路径

    # 如果路径不存在则创建路径
    if not os.path.exists(TEMP_RESULT_PATH):
        os.makedirs(TEMP_RESULT_PATH)
    if not os.path.exists(OUT_JSON_PATH):
        os.makedirs(OUT_JSON_PATH)

    # 解析参数
    # args = parse_args()
    # if args.config is not None:
    #     CONFIG_FILE = args.config
    # if args.checkpoint is not None:
    #     CHECKPOINT_FILE = args.checkpoint
    # if args.img_input is not None:
    #     KFB_PATH = args.img_input
    # if args.img_output is not None:
    #     OUT_JSON_PATH = args.img_output

    SCALE       = 20          # 缩放尺度
    window_size = 600         # 滑窗大小(w, h)
    step_size   = 500         # 滑窗步进(dx, dy)
    checkpoint  = 0           # 从检查点开始，检查点为上一次最后生成的json文件的顺序索引

    # 初始化模型
    print('=== initialing detector ===')
    # model = init_detector(CONFIG_FILE, CHECKPOINT_FILE)
    ## copy from detect.py
    parser = argparse.ArgumentParser()
    #parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.7, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    #parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    model.eval()  # Set in evaluation mode
    

    # 推断图片
    print('=== inference start ===')
    kfbread = kfbReader.reader() # 创建阅读器对象
    kfb_list = os.listdir(KFB_PATH)
    kfb_list.sort(key=lambda x: int(x[6:-4]))   # 按文件名排序
    
    # 遍历kfb文件
    kfb_cnt = checkpoint
    kfb_total = len(kfb_list)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    for kfb_fileName in kfb_list[checkpoint:]:
        kfb_cnt += 1
        # 读取kfb文件
        kfb_fullFileName = os.path.join(KFB_PATH, kfb_fileName)
        kfbread.ReadInfo(kfb_fullFileName, SCALE, True)    # 读取kfb文件信息(必须的)
        fullImg_width = kfbread.getWidth()                 # 读取全图的宽度
        fullImg_heigth = kfbread.getHeight()               # 读取全图的高度
        
        dataloader = DataLoader(
            ImageFolder(kfbread, winSize=window_size, imgSize=opt.img_size, stepSize=step_size),
            batch_size=opt.batch_size,
            shuffle=False, 
            num_workers=opt.n_cpu,
        )
        # 滑动窗口检测
        bboxes_list = []
        # bbox_dict = {}    # 错误定义的地方
        window_cnt = 0
        #window_nums = ((fullImg_width-window_size[0]+step_size[0])//step_size[0]) * ((fullImg_heigth-window_size[1]+step_size[1])//step_size[1])
        #barPrefix = '('+str(kfb_cnt)+'/'+str(kfb_total)+')...' + kfb_fileName
        # bar = progressbar.ProgressBar(prefix=barPrefix, max_value=window_nums).start()  
        for batch_i, (xs, ys, input_imgs) in enumerate(dataloader):
            # Get detections
            # print(input_imgs.shape)
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            
            # print(len(detections))
            for i, detection in enumerate(detections):
                if detection is not None: #.shape[0] > 0:
                    # 存储结果
                    # print(detection.shape)
                    detection = detection.tolist()
                    x, y = xs[i].item(), ys[i].item()
                    for xmin, ymin, xmax, ymax, conf, cls_conf, cls_pred in detection:
                        # print(xmin, ymin, xmax, ymax, conf, cls_conf, cls_pred)
                        bbox_dict = {}  # 必须定义在这里(每次进入for循环都是实例化新的字典对象, 每次append的元素都指向新的对象),
                                        # 不能定义在前面(实质只有一个实例化的字典对象, 导致append的每一个元素都指向同一个对象)
                        bbox_dict["x"] = int(round(x + xmin))
                        bbox_dict["y"] = int(round(y + ymin))
                        bbox_dict["w"] = int(round(xmax - xmin))
                        bbox_dict["h"] = int(round(ymax - ymin))
                        bbox_dict["p"] = round(float(conf), 5)
                        # print(bbox_dict)
                        bboxes_list.append(bbox_dict)
            # 写入json
            json_fileName = kfb_fileName.split('.')[0] + '.json'
            json_filePath = os.path.join(OUT_JSON_PATH, json_fileName)
            with open(json_filePath, 'w') as outfile:  
                outfile.write(json.dumps(bboxes_list))


if __name__ == '__main__':
    main()