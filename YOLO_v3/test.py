from matplotlib import patches
import paddle
from data.dataset import cocoDetectionDataset
from pycocotools.coco import COCO
import numpy as np

# path = '../data/coco/labels/val2014/COCO_val2014_000000000208.txt'
# boxes = np.loadtxt(path)
# boxes = paddle.to_tensor(boxes)
# print(paddle.reshape(boxes,(-1,5)))

data_path = '../data/coco/5k.txt'
data_set = cocoDetectionDataset(data_path)

# for i in range(10):
#     print("<< {} <<".format(i))
#     _, img, label = data_set.__getitem__(i)
#     print(img.shape)
#     print(label.shape)
#     print(">> {} >>".format(i))

# print(data_set.__len__)
# x1 = paddle.zeros((3,416,416))
# x2 = paddle.zeros((3,416,416))
# x3 = paddle.zeros((3,416,416))
# out = paddle.stack([x1, x2, x3], axis=0)
# print(out.shape)  # [3, 1, 2]
# print(out)
test_loader = paddle.io.DataLoader(data_set,
                                   batch_size=2,
                                   shuffle=True,
                                   collate_fn=data_set.collate_fn)

for i, (_, image, label) in enumerate(test_loader()):
    print("<< {} <<".format(i))
    print(image.shape)
    print(label.shape)
    print(">> {} >>".format(i))
