import paddle
import tqdm
from paddle.vision.transforms import ToTensor
# 启动单机多卡训练
import paddle.distributed as dist 
from paddle.distributed.spawn import spawn
from paddle.fluid.dygraph.parallel import DataParallel
import numpy as np
from argparse import ArgumentParser

from model import Mnist
from loss import crossEntropyLoss

def parse_args():
    parser = ArgumentParser(description="Train a mnist classifier!")
    parser.add_argument('-e', '--epochs', type=int, default=6,
                        help='number of epochs for training')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='number of examples for each iteration')
    parser.add_argument('--resume', '-r',action='store_true', default=False,
                        help='resume from checkpoint')
    return parser.parse_args()


def main():

    print("\n Paddlepaddle version: {}\n".format(paddle.__version__))

    args = parse_args()

    # 初始化并行环境
    # dist.init_parallel_env()

    # 加载数据集
    train_dataset = paddle.vision.datasets.MNIST(mode='train',transform=ToTensor())
    val_dataset = paddle.vision.datasets.MNIST(mode='test',transform=ToTensor())

    train_loader = paddle.io.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    test_loader = paddle.io.DataLoader(val_dataset,batch_size=args.batch_size)

    # 模型搭建
    mnist = Mnist()
    paddle.summary(net=mnist,input_size=(-1,1,28,28))
    # 增加paddle.DataParallel封装
    # mnist = paddle.DataParallel(mnist)

    optim = paddle.optimizer.Adam(parameters=mnist.parameters())
    loss_fn = paddle.nn.CrossEntropyLoss()

    start_epoch = 0
    epochs = args.epochs

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        info = np.load('./weights/info.npy',allow_pickle=True).item()
        start_epoch = info['epoch'] + 1
        val_loss = info['loss']
        val_acc = info['acc']
        print('Epoch {}, validation loss is: {loss:.4f}, validation accuracy is {acc:.4f}\n'\
            .format(start_epoch,loss=val_loss,acc=val_acc))
        mnist_state_dict = paddle.load('./weights/mnist.pdparams')
        mnist.set_state_dict(mnist_state_dict)
        optim_state_dict = paddle.load('./weights/optim.pdopt')
        optim.set_state_dict(optim_state_dict)


    for epoch in range(start_epoch,epochs):
        # 训练
        mnist.train()
        loader = tqdm.tqdm(train_loader)
        for batch_id, (image,label) in enumerate(loader):
            predicts = mnist(image)
            loss = loss_fn(predicts,label)
            acc = paddle.metric.accuracy(predicts,label)
            loss.backward()
            optim.step()
            optim.clear_grad()          
            description = ('Epoch {} (loss: {loss:.4f}, acc: {acc:.4f})'
                           .format(epoch, loss=loss.numpy().item(),acc=acc.numpy().item()))
            loader.set_description(description)

        # 测试
        mnist.eval()
        losses = 0.0
        accuracy = 0.0
        count = 0
        for batch_id, (image,label) in enumerate(test_loader):
            predicts = mnist(image)
            loss = loss_fn(predicts,label)
            acc = paddle.metric.accuracy(predicts,label)
            count += 1
            losses += loss.numpy().item() 
            accuracy += acc.numpy().item() 
        print("Testing: loss:{loss:.4f}, acc: {acc:.4f}".format(loss=losses/count,acc=accuracy/count))

        # 保存参数
        print('Saving checkpoint..')
        state = {
            'epoch': epoch,
            'loss':losses/count,
            'acc':accuracy/count
        }
        # 目前仅支持存储 Layer 或者 Optimizer 的 state_dict 。
        np.save('./weights/info.npy',state,allow_pickle=True) # 保存相关参数
        paddle.save(mnist.state_dict(),'./weights/mnist.pdparams')
        paddle.save(optim.state_dict(),'./weights/optim.pdopt')
        

if __name__ == '__main__':
    main()
    print('Done!')
    # dist.spawn(main) #使用当前所有可见GPU并行训练


    


