import paddle
import argparse
import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.vision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy mnist classifier with puthon.")
    parser.add_argument("--model_file",type=str,default='./deploy/mnist.pdmodel',
                        help="model filename.")
    parser.add_argument("--params_file",type=str,default='./deploy/mnist.pdiparams',
                        help="parameter filename.")
    parser.add_argument("--batch_size",type=int,default=1,
                        help="batch size.")
    return parser.parse_args()

def main():
    args = parse_args()

    # 配置
    config = Config(args.model_file,args.params_file)
    config.disable_gpu()
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)

    # 创建paddlePredictor
    predictor = create_predictor(config)

    # 获取输入
    val_dataset = paddle.vision.datasets.MNIST(mode='test',transform=transforms.ToTensor())
    (image,label) = val_dataset[np.random.randint(10000)]
    # fake_input = np.random.randn(1, 1, 28, 28).astype("float32")
    # image = np.asndarray(image).astype("float32")
    # print(image.shape)
    image = image.numpy().reshape([1,1,28,28])
    # print(image.shape)
    # print(fake_input.shape)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    input_handle.reshape([1,1,28,28])
    input_handle.copy_from_cpu(image)


    # 运行predictor
    predictor.run()

    # 获取输出
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output = output_handle.copy_to_cpu()

    print("True label: ",label.item())
    print("Prediction: ",np.argmax(output))

if __name__=='__main__':
    main()