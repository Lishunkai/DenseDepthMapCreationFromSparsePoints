import os
import os.path
import cv2 # opencv的python版本
import numpy as np
import torch.utils.data as data
import h5py # HDF5的python版本
import transforms
from PIL import Image

IMG_EXTENSIONS = [
    '.h5',
]

def is_image_file(filename):
    # return any(): 只要迭代器中有一个元素为真就返回为真。
    # 详见https://blog.csdn.net/heatdeath/article/details/70178511
    # 详见https://blog.csdn.net/u013630349/article/details/47374333
    # 其中，...for...in...是迭代器(iterable)
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    # ...for...in...if... 挑选出in后面的内容中符合if条件的元素，组成一个新的list
    # 详见http://www.jb51.net/article/86987.htm
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort() # 升序排列
    # Python中[]，()，{}的区别:
    # {}表示字典，[]表示数组，()表示元组
    # 数组的值可以改变，可以使用切片获取部分数据
    # 元组的值一旦设置，不可更改，不可使用切片
    # 详见https://zhidao.baidu.com/question/484920124.html
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

# 这个函数可以用于生成自己的数据集
def make_dataset(dir, class_to_idx):
    images = [] # 存放图片序号的数组
    # dir是路径
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)): # sorted(): 输出排序后的列表 升序排列
        # print(target)
        # target只是文件名，不含路径。为了获得文件的完整路径，用os.path.join(dirpath, name)
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)): # tuple: 元组，数组
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item) # append()用于在列表末尾添加新的对象，和C++中的push_back()一样

    return images


def h5_loader(path):
    # 关于HDF5的文件操作，详见https://blog.csdn.net/yudf2010/article/details/50353292
    h5f = h5py.File(path, "r") # r是读的意思
    rgb = np.array(h5f['rgb']) # 使用array()函数可以将python的array_like数据转变成数组形式，使用matrix()函数转变成矩阵形式。
    # 基于习惯，在实际使用中较常用array而少用matrix来表示矩阵。
    rgb = np.transpose(rgb, (1, 2, 0)) # 关于np.transpose()对高维数组的转置，详见https://www.cnblogs.com/sunshinewang/p/6893503.html
    depth = np.array(h5f['depth'])

    return rgb, depth

iheight, iwidth = 480, 640 # raw image size
oheight, owidth = 228, 304 # image size after pre-processing
color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

# 数据增强（论文中的方法）
def train_transform(rgb, depth):
    s = np.random.uniform(1.0, 1.5) # random scaling
    # print("scale factor s={}".format(s))
    depth_np = depth / s
    angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

    # perform 1st part of data augmentation
    transform = transforms.Compose([
        transforms.Resize(250.0 / iheight), # this is for computational efficiency, since rotation is very slow
        transforms.Rotate(angle),
        transforms.Resize(s),
        transforms.CenterCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ])
    rgb_np = transform(rgb)

    # random color jittering 
    rgb_np = color_jitter(rgb_np)

    rgb_np = np.asfarray(rgb_np, dtype='float') / 255
    depth_np = transform(depth_np)

    return rgb_np, depth_np

# 数据增强（论文中的方法）
def val_transform(rgb, depth):
    depth_np = depth

    # perform 1st part of data augmentation
    transform = transforms.Compose([
        transforms.Resize(240.0 / iheight),
        transforms.CenterCrop((oheight, owidth)),
    ])
    rgb_np = transform(rgb)
    rgb_np = np.asfarray(rgb_np, dtype='float') / 255
    depth_np = transform(depth_np)

    return rgb_np, depth_np

def rgb2grayscale(rgb):
    return rgb[:,:,0] * 0.2989 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.114

# 仅限RGB图像
def MatrixTocvMat(data):
    data = data*255
    im = Image.fromarray(data.astype(np.uint8))
    new_im = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)  
    return new_im


to_tensor = transforms.ToTensor()

class NYUDataset(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd']

    def __init__(self, root, type, modality='rgb', num_samples=0, loader=h5_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        if type == 'train':
            self.transform = train_transform
        elif type == 'val':
            self.transform = val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))
        self.loader = loader

        if modality in self.modality_names: # 如果在...中有...
            self.modality = modality
            if modality in ['rgbd', 'd', 'gd']:
                if num_samples <= 0:
                    raise (RuntimeError("Invalid number of samples: {}\n".format(num_samples)))
                self.num_samples = num_samples
            else:
                self.num_samples = 0
        else:
            raise (RuntimeError("Invalid modality type: " + modality + "\n"
                                "Supported dataset types are: " + ''.join(self.modality_names)))

	# 生成稀疏深度图 原版
    def create_sparse_depth(self, depth, num_samples):
        prob = float(num_samples) / depth.size # 概率
        mask_keep = np.random.uniform(0, 1, depth.shape) < prob # 生成一个0-1的mask，0-1是随机产生的，0-1产生的概率小于预设的概率
        sparse_depth = np.zeros(depth.shape)
        sparse_depth[mask_keep] = depth[mask_keep] # 把深度图中和mask_keep对应的元素赋值给sparse_depth中的对应元素
        return sparse_depth

    # 生成稀疏深度图 ORB特征提取
    def create_sparse_depth_ORB(self, rgb_np, depth, num_samples):
        # num_samples = int(prob * depth.size)
        rgb = MatrixTocvMat(rgb_np)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(num_samples) # 这里没有网格化
        kp = orb.detect(gray, None)
        print("number of ORB KeyPoints:", len(kp))

        # print(len(kp)) # keypoint的个数
        # print(kp[0]) # 多少个角点，就有多少个下标
        # tu = kp[0].pt   #（提取坐标） pt指的是元组 tuple(x,y)
        # print(tu[0],tu[1]) # 输出第一个keypoint的x,y坐标

        sparse_depth = np.zeros(depth.shape)
        for i in range(len(kp)):
            tu = kp[i].pt
            x = int(tu[0])
            y = int(tu[1])
            sparse_depth[y,x] = depth[y,x] # x,y和行,列的顺序不一样
        
        return sparse_depth

    def create_rgbd(self, rgb, depth, num_samples):
        sparse_depth = self.create_sparse_depth_ORB(rgb, depth, num_samples)
        # sparse_depth = self.create_sparse_depth(depth, num_samples)
        
        # rgbd = np.dstack((rgb[:,:,0], rgb[:,:,1], rgb[:,:,2], sparse_depth))
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2) # append()相当于push_back()
        return rgbd

    # 获取原始图像
    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        rgb, depth = self.loader(path) # 这个loader就是h5_loader
        return rgb, depth

    def __get_all_item__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input_tensor, depth_tensor, input_np, depth_np) 
        """
        rgb, depth = self.__getraw__(index)
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth) # 经过数据增强步骤后的结果
        else:
            raise(RuntimeError("transform not defined"))

        # color normalization
        # rgb_tensor = normalize_rgb(rgb_tensor)
        # rgb_np = normalize_np(rgb_np)
        
        if self.modality == 'rgb':
            input_np = rgb_np
        elif self.modality == 'rgbd':
            input_np = self.create_rgbd(rgb_np, depth_np, self.num_samples)
        elif self.modality == 'd':
            input_np = self.create_sparse_depth_ORB(rgb_np, depth_np, self.num_samples)
            # input_np = self.create_sparse_depth(depth_np, self.num_samples)

        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor, input_np, depth_np

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input_tensor, depth_tensor) 
        """
        input_tensor, depth_tensor, input_np, depth_np = self.__get_all_item__(index)

        return input_tensor, depth_tensor

    def __len__(self):
        return len(self.imgs)