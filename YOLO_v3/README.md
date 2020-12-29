# README

本项目基于Paddlepaddle 2.0实现YOLO v3。

## 1. 数据集准备

YOLO v3使用COCO数据集进行训练，本节进行数据集准备工作。COCO数据集提供了用于加载、解析和可视化的API，现在进行安装：

```python
# 安装必要编译套件
$ sudo apt install build-essential
# 新建conda环境
$ conda create -n yolo_v3 python=3.7
# 激活环境
$ conda activate yolo_v3
# 安装cocoapi
$ pip install pycocotools
```

下载图像数据集（train2017.zip,val2017.zip,test2017.zip）并解压到 cocoapi/images 文件夹，下载标签数据集（）并解压到 cocoapi/annotations 文件夹：

```python
# 新建数据集存放目录，此处为了方便多个项目共同访问，新建与YOLO_v3同级的data目录
$ cd ../
$ mkdir data
$ cd data
$ mkdir coco
$ cd coco
#下载图像数据
$ mkdir images
$ cd images
$ wget -c https://pjreddie.com/media/files/train2014.zip 
$ wget -c https://pjreddie.com/media/files/val2014.zip
# 解压数据集
$ unzip train2014.zip  
$ unzip val2014.zip
#下载标签数据
$ cd ..
$ makdir annotations
$ cd annotations
$ wget -c https://pjreddie.com/media/files/instances_train-val2014.zip
$ unzip annotations_trainval2014.zip
# 最后，数据集实在下不下来，就用迅雷下载好再拷贝过来吧，迅雷速度还行
```
最终数据集目录结构如下：
```python
--data
    |__cocoa
    |      |__images
    |      |__annotations
    |__...
```
coco 数据集的标注数据采用json格式存储，基本结构如下：


```python
{
    "info": info,
    "images": [image], #这是一个image的列表，如[image1,image2,....]，
    "annotations": [annotation], #同理
    "licenses": [license], #同理
    "categories": [category]
}

info{
    "year": int, 
    "version": str, 
    "description": str, 
    "contributor": str, 
    "url": str, 
    "date_created": datetime,
}

image{
    "id": int, 
    "width": int, 
    "height": int, 
    "file_name": str, 
    "license": int, 
    "flickr_url": str, 
    "coco_url": str, 
    "date_captured": datetime,
}

license{
    "id": int, 
    "name": str, 
    "url": str,
}


# 目标检测的annotation
annotation{
    "id": int, 
    "image_id": int, 
    "category_id": int, 
    "segmentation": RLE or [polygon], 
    "area": float, 
    "bbox": [x,y,width,height], 
    "iscrowd": 0 or 1,
}

categories[{
    "id": int, 
    "name": str, 
    "supercategory": str,
}]
```
## 2. 安装依赖

