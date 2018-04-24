# 画图和保存图片的函数

import numpy as np
import matplotlib.pyplot as plt # 导入画图工具
from PIL import Image # Python Imaging Library

cmap = plt.cm.jet # cmap: 颜色图谱（colormap) 详见https://blog.csdn.net/haoji007/article/details/52063168

def merge_into_row(input, target, depth_pred):
    # np.squeeze(): 从数组的形状中删除单维条目，即把shape中为1的维度去掉
    # 对于高维数组，transpose需要用到一个由轴编号组成的元组，才能进行转置。详见https://www.cnblogs.com/sunshinewang/p/6893503.html
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth = np.squeeze(target.cpu().numpy())
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:,:,:3] # H, W, C [:,:,:3]是什么意思？
    pred = np.squeeze(depth_pred.data.cpu().numpy())
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    pred = 255 * cmap(pred)[:,:,:3] # H, W, C
    img_merge = np.hstack([rgb, depth, pred]) # 将一系列数组按输入顺序水平地排成一排
    
    # img_merge.save(output_directory + '/comparison_' + str(epoch) + '.png')
    return img_merge

def add_row(img_merge, row):
    return np.vstack([img_merge, row]) # 将一系列数组按输入顺序竖直地排成一列

def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8')) # 设置保存的数据格式
    img_merge.save(filename)