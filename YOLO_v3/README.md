# README

本项目基于Paddlepaddle 2.0实现YOLO v3。

## 1. 数据集准备

YOLO v3使用COCO数据集进行训练，本节进行数据集准备工作。COCO数据集提供了用于加载、解析和可视化的API，现在进行安装：

```python
# 新建数据集存放目录，这里为了方便多个项目共同使用coco数据集，新建与YOLO v3平行的data目录
cd ..
mkdir data
cd data
# 下载coco api并编译安装
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make -j4 install 
# Windows 平台安装coco api参考 https://github.com/philferriere/cocoapi.git
```

下载图像数据集（train2017.zip,val2017.zip,test2017.zip）并解压到 cocoapi/images 文件夹，下载标签数据集（）并解压到 cocoapi/annotations 文件夹：

```python
#下载图像数据

mkdir images
cd images

wget -c http://images.cocodataset.org/zips/train2017.zip  # --continue, 断点续传
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip

unzip train2017.zip  
unzip val2017.zip
unzip test2017.zip

#下载标签数据

cd ..
makdir annotations
cd annotations

wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip annotations_trainval2017.zip
```
最终数据集目录结构如下：
```python
--data
    |__cocoapi
          |__images
          |__annotations
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

