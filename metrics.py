# 这个脚本文件中定义了许多评价模型的标准

import torch
import math
import numpy as np

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

# 新式类，详见https://www.cnblogs.com/liulipeng/p/7069004.html
class Result(object):
    # 带有两个下划线开头的函数是声明该属性为私有,不能在类的外部被使用或直接访问
    # init函数（方法）支持带参数的类的初始化，也可为声明该类的属性。第一个参数必须是self，后续参数则可以自由指定。
    # 在类的内部，使用def关键字可以为类定义一个函数（方法）。类方法必须包含参数self,且为第一个参数。
    # python函数只能先定义再调用
    # Python中的self等价于C++中的self指针和Java、C#中的this参数。
    # self指的是传入的实例(instance)，不同实例类的属性值不同以及方法执行结果不同
    # 详见https://blog.csdn.net/ly_ysys629/article/details/54893185
    def __init__(self):
        # 一行读取多个值
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self): # 将最坏的情况初始化为无穷大
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        valid_mask = target>0 # valid_mask为ture或false
        output = output[valid_mask] # [ ]:代表list列表数据类型,列表是一种可变的序列
        target = target[valid_mask]

        abs_diff = (output - target).abs()

        self.mse = (torch.pow(abs_diff, 2)).mean() # 均方误差     torch.pow()和python本身的**有什么区别？
        self.rmse = math.sqrt(self.mse) # 均方根误差
        self.mae = abs_diff.mean() # Mean Absolute Error
        self.lg10 = (log10(output) - log10(target)).abs().mean()
        self.absrel = (abs_diff / target).mean()

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = (maxRatio < 1.25).float().mean()
        self.delta2 = (maxRatio < 1.25 ** 2).float().mean()
        self.delta3 = (maxRatio < 1.25 ** 3).float().mean()
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = abs_inv_diff.mean()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3
        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time

    def average(self):
        avg = Result() # Result()是个类，这句话是定义一个Result()类的对象avg
        # 调用avg对象中的update函数
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count, 
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count)
        return avg