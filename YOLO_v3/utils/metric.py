import paddle

class SelfDefineMetric(paddle.metric.Metric):
    """
    1. 继承paddle.metric.Metric
    """
    def __init__(self):
        """
        2. 构造函数实现，自定义参数即可
        """
        super(SelfDefineMetric, self).__init__()

    def name(self):
        """
        3. 实现name方法，返回定义的评估指标名字
        """
        return '自定义评价指标的名字'

    def compute(self, ...)
        """
        4. 本步骤可以省略，实现compute方法，这个方法主要用于`update`的加速，可以在这个方法中调用一些paddle实现好的Tensor计算API，编译到模型网络中一起使用低层C++ OP计算。
        """
        return 自己想要返回的数据，会做为update的参数传入。

    def update(self, ...):
        """
        5. 实现update方法，用于单个batch训练时进行评估指标计算。
        - 当`compute`类函数未实现时，会将模型的计算输出和标签数据的展平作为`update`的参数传入。
        - 当`compute`类函数做了实现时，会将compute的返回结果作为`update`的参数传入。
        """
        return acc value

    def accumulate(self):
        """
        6. 实现accumulate方法，返回历史batch训练积累后计算得到的评价指标值。
        每次`update`调用时进行数据积累，`accumulate`计算时对积累的所有数据进行计算并返回。
        结算结果会在`fit`接口的训练日志中呈现。
        """
        # 利用update中积累的成员变量数据进行计算后返回
        return accumulated acc value

    def reset(self):
        """
        7. 实现reset方法，每个Epoch结束后进行评估指标的重置，这样下个Epoch可以重新进行计算。
        """
        # do reset action


from paddle.metric import Metric

class Precision(Metric):
    """
    Precision (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances. Refer to
    https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
    Noted that this class manages the precision score only for binary
    classification task.

    ......
    """

    def __init__(self, name='precision', *args, **kwargs):
        super(Precision, self).__init__(*args, **kwargs)
        self.tp = 0  # true positive
        self.fp = 0  # false positive
        self._name = name

    def update(self, preds, labels):
        """
        Update the states based on the current mini-batch prediction results.
        Args:
            preds (numpy.ndarray): The prediction result, usually the output
               of two-class sigmoid function. It should be a vector (column
               vector or row vector) with data type: 'float64' or 'float32'.
           labels (numpy.ndarray): The ground truth (labels),
               the shape should keep the same as preds.
               The data type is 'int32' or 'int64'.
        """
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        elif not _is_numpy_(preds):
            raise ValueError("The 'preds' must be a numpy ndarray or Tensor.")
        if isinstance(labels, paddle.Tensor):
            labels = labels.numpy()
        elif not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray or Tensor.")

        sample_num = labels.shape[0]
        preds = np.floor(preds + 0.5).astype("int32")

        for i in range(sample_num):
            pred = preds[i]
            label = labels[i]
            if pred == 1:
                if pred == label:
                    self.tp += 1
                else:
                    self.fp += 1

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.tp = 0
        self.fp = 0

    def accumulate(self):
        """
        Calculate the final precision.

        Returns:
           A scaler float: results of the calculated precision.
        """
        ap = self.tp + self.fp
        return float(self.tp) / ap if ap != 0 else .0

    def name(self):
        """
        Returns metric name
        """
        return self._name