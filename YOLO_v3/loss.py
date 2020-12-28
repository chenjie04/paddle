import paddle 

class crossEntropyLoss(paddle.nn.Layer):
   """
   1. 继承paddle.nn.Layer
   """
   def __init__(self):
       """
       2. 构造函数根据自己的实际算法需求和使用需求进行参数定义即可
       """
       super(crossEntropyLoss, self).__init__()

   def forward(self, input, label):
       """
       3. 实现forward函数，forward在调用时会传递两个参数：input和label
           - input：单个或批次训练数据经过模型前向计算输出结果
           - label：单个或批次训练数据对应的标签数据
           接口返回值是一个Tensor，根据自定义的逻辑加和或计算均值后的损失
       """
       # 使用Paddle中相关API自定义的计算逻辑
       # output = xxxxx
       # return output

class SoftmaxWithCrossEntropy(paddle.nn.Layer):
    def __init__(self):
       super(SoftmaxWithCrossEntropy, self).__init__()

    def forward(self, input, label):
       loss = F.softmax_with_cross_entropy(input,
                                           label,
                                           return_softmax=False,
                                           axis=1)
       return paddle.mean(loss)