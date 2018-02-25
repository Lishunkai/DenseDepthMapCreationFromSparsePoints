import torch
import torch.nn as nn
from torch.autograd import Variable

# L2范数
class MaskedMSELoss(nn.Module):
    # Module是pytorch提供的一个基类，每次我们搭建神经网络时都要继承这个类，继承这个类会使搭建网络的过程变得异常简单
    # 详见https://blog.csdn.net/u012436149/article/details/78281553
    def __init__(self):
        # super是继承父类(超类)的一种方法
        # 详见https://www.cnblogs.com/HoMe-Lin/p/5745297.html
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        # 在没完善一个程序之前，我们不知道程序在哪里会出错。与其让它在运行最后崩溃，不如在出现错误条件时就崩溃，这时就需要assert断言
        # 详见https://www.cnblogs.com/liuchunxiao83/p/5298016.html
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

# L1范数
class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
