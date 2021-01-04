# -----------------------------------------------------------------------------
# 数据预处理：
#       1. 清洗数据
#       2. 重新划分训练集与测试集。从val2014中选取5k张作为测试集，其余的加上train2014作为训练集。
#
# -----------------------------------------------------------------------------

from itertools import count
import os
import numpy as np
from pycocotools.coco import COCO

val_annFile = '../data/coco/annotations/instances_val2014.json'
val_coco = COCO(val_annFile)
val_ids = list(val_coco.imgs.keys())

train_annFile = '../data/coco/annotations/instances_train2014.json'
train_coco = COCO(train_annFile)
train_ids = list(train_coco.imgs.keys())

val_file = open('../data/coco/5k.txt', 'w+')
train_file = open('../data/coco/trainvalno5k.txt', 'w+')

# 处理val2014
count = 0
for i in range(len(val_ids)):
    if count <= 5000:
        img_id = val_ids[i]
        file_name = val_coco.loadImgs(img_id)[0]['file_name']
        annids = val_coco.getAnnIds(imgIds=img_id)
        anns = val_coco.loadAnns(annids)
        if len(anns) > 0:
            val_file.write('../data/coco/images/val2014/' + file_name + '\n')
            labels_path = '../data/coco/labels/val2014/'
            if not os.path.exists(labels_path):
                os.makedirs(labels_path)
            lebels_file = open(
                labels_path +
                file_name.replace(".png", ".txt").replace(".jpg", ".txt"),
                'w+')
            for i in range(len(anns)):
                lebels_file.write('%s %s %s %s %s\n' %
                                  (anns[i]['category_id'], anns[i]['bbox'][0],
                                   anns[i]['bbox'][1], anns[i]['bbox'][2],
                                   anns[i]['bbox'][3]))
            lebels_file.close()
            count += 1
        else:
            print('sikp the {}-th image, as it has not bbox...'.format(i))
            continue
    else:
        img_id = val_ids[i]
        file_name = val_coco.loadImgs(img_id)[0]['file_name']
        annids = val_coco.getAnnIds(imgIds=img_id)
        anns = val_coco.loadAnns(annids)
        if len(anns) > 0:
            train_file.write('../data/coco/images/train2014/' + file_name + '\n')
            labels_path = '../data/coco/labels/val2014/'
            if not os.path.exists(labels_path):
                os.makedirs(labels_path)
            lebels_file = open(
                labels_path +
                file_name.replace(".png", ".txt").replace(".jpg", ".txt"),
                'w+')
            for i in range(len(anns)):
                lebels_file.write('%s %s %s %s %s\n' %
                                  (anns[i]['category_id'], anns[i]['bbox'][0],
                                   anns[i]['bbox'][1], anns[i]['bbox'][2],
                                   anns[i]['bbox'][3]))
            lebels_file.close()
        else:
            print('sikp the {}-th image, as it has not bbox...'.format(i))
            continue
val_file.close()

# 处理train2014
for i in range(len(train_ids)):
    img_id = train_ids[i]
    file_name = train_coco.loadImgs(img_id)[0]['file_name']
    annids = train_coco.getAnnIds(imgIds=img_id)
    anns = train_coco.loadAnns(annids)
    if len(anns) > 0:
        train_file.write('../data/coco/images/train2014/' + file_name + '\n')
        labels_path = '../data/coco/labels/train2014/'
        if not os.path.exists(labels_path):
            os.makedirs(labels_path)
        lebels_file = open(
            labels_path +
            file_name.replace(".png", ".txt").replace(".jpg", ".txt"), 'w+')
        for i in range(len(anns)):
            lebels_file.write(
                '%s %s %s %s %s\n' %
                (anns[i]['category_id'], anns[i]['bbox'][0],
                 anns[i]['bbox'][1], anns[i]['bbox'][2], anns[i]['bbox'][3]))
        lebels_file.close()
    else:
        print('sikp the {}-th image, as it has not bbox...'.format(i))
        continue

train_file.close()

print('Done!')