import paddle
from dataset import cocoDetectionDataset

val_dataDir = '../data/coco/images/val2014'
val_annFile = '../data/coco/annotations/instances_val2014.json'
val_dataset = cocoDetectionDataset(root=val_dataDir,annFile=val_annFile)

for i in range(300):
    print("<< {} <<".format(i))
    img, label = val_dataset.__getitem__(i)
    print(img.shape)
    print(label.shape)
    print(">> {} >>".format(i))


# test_loader = paddle.io.DataLoader(val_dataset,batch_size=2,shuffle=True)

# for _, (image, label) in enumerate(test_loader()):
#      print(image.shape)
#      print(label.shape)