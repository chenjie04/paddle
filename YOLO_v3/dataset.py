import os
from PIL import Image
import numpy as np
import cv2
import paddle
from paddle.io import Dataset
from pycocotools.coco import COCO 


class cocoDetectionDataset(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        
    """
    def __init__(self, root, annFile,img_size=416):
        super(cocoDetectionDataset, self).__init__()
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.img_size = img_size

        COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                           9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                          18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                          27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                          37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                          46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                          54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                          62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                          74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                          82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
        self.label_map = COCO_LABEL_MAP


    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]

        # 处理图像，返回一张416x416的图像
        path = coco.loadImage(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root,path)).convert('RGB')
        img = paddle.vision.transforms.ToTensor()(img)
        # Pad to square resolution
        c, h, w = img.shape()
        dim_diff = np.abs(h-w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding pad =（左，右，上，下）
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding
        img = paddle.nn.functional.pad(img,pad,"constant",value=0.0)
        _, padded_h, padded_w = img.shape
        # resize
        # unsqueeze(0)在前面增加一个维度，squeeze(0)去掉第一个维度
        img = paddle.nn.functional.interpolate(img.unsqueeze(0), size=self.img_size, mode="nearest").squeeze(0)

        # 处理图像对应目标检测标注信息
        
        return img, targets

    def __len__(self):
        return len(self.ids)