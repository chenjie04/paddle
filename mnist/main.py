from paddle.distributed.spawn import spawn
from paddle.fluid.dygraph.parallel import DataParallel
import paddle
print("Paddlepaddle version: ",paddle.__version__)
import tqdm
from paddle.vision.transforms import ToTensor
# 启动单机多卡训练
# import paddle.distributed as dist 

from model import Mnist

def main():
    # 初始化并行环境
    # dist.init_parallel_env()

    # 加载数据集
    train_dataset = paddle.vision.datasets.MNIST(mode='train',transform=ToTensor())
    val_dataset = paddle.vision.datasets.MNIST(mode='test',transform=ToTensor())

    train_loader = paddle.io.DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = paddle.io.DataLoader(val_dataset,batch_size=64)

    # 模型搭建
    mnist = Mnist()
    # 增加paddle.DataParallel封装
    # mnist = paddle.DataParallel(mnist)

    optim = paddle.optimizer.Adam(parameters=mnist.parameters())
    loss_fn = paddle.nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(epochs):
        mnist.train()
        loader = tqdm.tqdm(train_loader)
        for batch_id, data in enumerate(loader):
            data_x = data[0]
            labels = data[1]
            predicts = mnist(data_x)
            loss = loss_fn(predicts,labels)
            acc = paddle.metric.accuracy(predicts,labels)
            loss.backward()
            optim.step()
            optim.clear_grad()          
            description = ('Epoch {} (loss: {loss:.4f}, acc: {acc:.4f})'
                           .format(epoch, loss=loss.numpy()[0],acc=acc.numpy()[0]))
            loader.set_description(description)

        mnist.eval()
        losses = 0.0
        accuracy = 0.0
        count = 0
        for batch_id, data in enumerate(test_loader):
            data_x = data[0]
            labels = data[1]
            predicts = mnist(data_x)
            loss = loss_fn(predicts,labels)
            acc = paddle.metric.accuracy(predicts,labels)
            count += 1
            losses += loss.numpy()[0] 
            accuracy += acc.numpy()[0] 
        print("Testing: loss:{loss:.4f}, acc: {acc:.4f}".format(loss=losses/count,acc=accuracy/count))
        

if __name__ == '__main__':
    main()
    # dist.spawn(main) #使用当前所有可见GPU并行训练


    


