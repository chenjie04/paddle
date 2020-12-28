# Readme

本项目基于Paddlepaddle 2.0实现YOLO v3。

## 数据集准备

YOLO v3使用COCO数据集进行训练，本节进行数据集准备工作。COCO数据集提供了用于加载、解析和可视化的API，现在进行安装：

```shell
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
