# 导入一般的文件操作模块
import argparse # 导入自动命令解析器 详见https://blog.csdn.net/ali197294332/article/details/51180628
import os # 导入Python的系统基础操作模块
import shutil # 导入高级的文件操作模块 详见https://www.cnblogs.com/MnCu8261/p/5494807.html
import time # 导入对时间操作的函数 详见http://www.jb51.net/article/87721.htm
import sys # 导入系统相关的信息模块
import csv # 导入处理csv格式文件的相关模块 详见https://www.cnblogs.com/yanglang/p/7126660.html

# 导入pytorch相关的模块
import torch
import torch.nn as nn
import torch.nn.parallel # 可以实现模块级别的并行计算。可以将一个模块forward部分分到各个gpu中，然后backward时合并gradient到original module。
import torch.backends.cudnn as cudnn # cudnn: CUDA Deep Neural Network 相比标准的cuda，它在一些常用的神经网络操作上进行了性能的优化，比如卷积，pooling，归一化，以及激活层等等。
import torch.optim
import torch.utils.data

# 导入自定义的模块
from nyu_dataloader import NYUDataset
from models import Decoder, ResNet
from metrics import AverageMeter, Result
import criteria
import utils

model_names = ['resnet18', 'resnet50']
loss_names = ['l1', 'l2']
data_names = ['NYUDataset']
decoder_names = Decoder.names
modality_names = NYUDataset.modality_names

cudnn.benchmark = True

# 创建一个解析处理器
parser = argparse.ArgumentParser(description='Sparse-to-Dense Training')
# parser.add_argument('--data', metavar='DIR', help='path to dataset',
#                     default="data/NYUDataset")

# 设置多个参数
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
# metavar:占位字符串，用于在输出帮助信息时，代替当前命令行选项的附加参数的值进行输出
# join：连接字符串数组。将字符串、元组、列表中的元素以指定的字符(分隔符)连接生成一个新的字符串。详见https://blog.csdn.net/zmdzbzbhss123/article/details/52279008
parser.add_argument('--data', metavar='DATA', default='nyudepthv2',
                    choices=data_names,
                    help='dataset: ' +
                        ' | '.join(data_names) +
                        ' (default: nyudepthv2)')
parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
                    choices=modality_names,
                    help='modality: ' +
                        ' | '.join(modality_names) +
                        ' (default: rgb)')
parser.add_argument('-s', '--num-samples', default=0, type=int, metavar='N',
                    help='number of sparse depth samples (default: 0)')
parser.add_argument('--decoder', '-d', metavar='DECODER', default='deconv2',
                    choices=decoder_names,
                    help='decoder: ' +
                        ' | '.join(decoder_names) +
                        ' (default: deconv2)')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', 
                    choices=loss_names,
                    help='loss function: ' +
                        ' | '.join(loss_names) +
                        ' (default: l1)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=True, help='use ImageNet pre-trained weights (default: True)')

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae', 
                'delta1', 'delta2', 'delta3', 
                'data_time', 'gpu_time']

# Result()是criteria.py中的类
best_result = Result()
best_result.set_to_worst()

# 定义函数
# 基本格式：
# def function_name(parameters):
#     expressions
# Python使用缩进来规定代码的作用域
def main():
    global args, best_result, output_directory, train_csv, test_csv # 全局变量
    args = parser.parse_args() # 获取参数值
    args.data = os.path.join('data', args.data)
# os.path.join()函数：将多个路径组合后返回
# 语法：os.path.join(path1[,path2[,......]])
# 注：第一个绝对路径之前的参数将被忽略
	# 注意if的语句后面有冒号
	# args中modality的参数值。modality之前定义过
    if args.modality == 'rgb' and args.num_samples != 0:
        print("number of samples is forced to be 0 when input modality is rgb")
        args.num_samples = 0
		# 若是RGB的sparse-to-dense，则在生成训练数据时将稀疏深度点设为0
  
    # create results folder, if not already exists
    output_directory = os.path.join('results',
        'NYUDataset.modality={}.nsample={}.arch={}.decoder={}.criterion={}.lr={}.bs={}'.
        format(args.modality, args.num_samples, args.arch, args.decoder, args.criterion, args.lr, args.batch_size)) # 输出文件名的格式

	# 如果路径不存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')
    
    # define loss function (criterion) and optimizer
    if args.criterion == 'l2':
        criterion = criteria.MaskedMSELoss().cuda() # 调用别的py文件中的内容时，若被调用的是函数，则直接写函数名即可；若被调用的是类，则要按这句话的格式写
        out_channels = 1
	# elif: else if
    elif args.criterion == 'l1':
        criterion = criteria.MaskedL1Loss().cuda()
        out_channels = 1

    # Data loading code
    print("=> creating data loaders ...")
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_dataset = NYUDataset(traindir, type='train', 
        modality=args.modality, num_samples=args.num_samples)
	# DataLoader是导入数据的函数
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    # set batch size to be 1 for validation
    val_dataset = NYUDataset(valdir, type='val', 
        modality=args.modality, num_samples=args.num_samples)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    print("=> data loaders created.")

    # evaluation mode
    if args.evaluate:
        best_model_filename = os.path.join(output_directory, 'model_best.pth.tar')
        if os.path.isfile(best_model_filename):
            print("=> loading best model '{}'".format(best_model_filename))
            checkpoint = torch.load(best_model_filename)
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        else: # else也要加:
            print("=> no best model found at '{}'".format(best_model_filename))
        validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
        return

    # optionally resume from a checkpoint
    elif args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']+1
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create new model
    else:
        # define model
        print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
        in_channels = len(args.modality) # len()返回对象的长度或项目个数
        if args.arch == 'resnet50':
            model = ResNet(layers=50, decoder=args.decoder, in_channels=in_channels,
                out_channels=out_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet18':
            model = ResNet(layers=18, decoder=args.decoder, in_channels=in_channels,
                out_channels=out_channels, pretrained=args.pretrained)
        print("=> model created.")

        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # create new csv files with only header
        # with open() as xxx: 的用法详见https://www.cnblogs.com/ymjyqsx/p/6554817.html
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:   
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    print(model)
    print("=> model transferred to GPU.")

    # for循环也要有:
    # 一般情况下，循环次数未知采用while循环，循环次数已知采用for
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        result, img_merge = validate(val_loader, model, epoch)
        # Python的return可以返回多个值

        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                # 字符串格式化输出
                # :3f中，3表示输出宽度，f表示浮点型。若输出位数小于此宽度，则默认右对齐，左边补空格。
                #       若输出位数大于宽度，则按实际位数输出。
                # :.3f中，.3表示指定除小数点外的输出位数，f表示浮点型。
                txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                    format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
            # None表示该值是一个空对象，空值是Python里一个特殊的值，用None表示。None不能理解为0，因为0是有意义的，而None是一个特殊的空值。
            # 你可以将None赋值给任何变量，也可以将任何变量赋值给一个None值的对象
            # None在判断的时候是False
            # NULL是空字符，和None不一样
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)

# Python中，万物皆对象，所有的操作都是针对对象的。一个对象包括两方面的特征：
# 属性：去描述它的特征
# 方法：它所具有的行为
# 所以，对象=属性+方法 （其实方法也是一种属性，一种区别于数据属性的可调用属性）

        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'model': model,
            'best_result': best_result,
            'optimizer' : optimizer,
        }, is_best, epoch)


def train(train_loader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time() # 计时开始
    # enumerate()用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中。
    for i, (input, target) in enumerate(train_loader):

        input, target = input.cuda(), target.cuda()
        # torch.autograd提供实现任意标量值功能的自动区分的类和功能。它将所有张量包装在Variable对象中。
        # Variable可以看作是对Tensor对象周围的一个薄包装，也包含了和张量相关的梯度，以及对创建它的函数的引用。
        # 此引用允许对创建数据的整个操作链进行回溯。需要BP的网络都是通过Variable来计算的。
        # pytorch中的所有运算都是基于Tensor的，Variable只是一个Wrapper，Variable的计算的实质就是里面的Tensor在计算。
        # 详见https://blog.csdn.net/KGzhang/article/details/77483383
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        torch.cuda.synchronize() # 等待当前设备上所有流中的所有内核完成
        data_time = time.time() - end # 计算用时

        # compute depth_pred
        end = time.time()
        depth_pred = model(input_var)
        loss = criterion(depth_pred, target_var)
        # optimizer包提供训练时更新参数的功能
        optimizer.zero_grad() # zero the gradient buffers，必须要置零
        # 在BP的时候，pytorch将Variable的梯度放在Variable对象中，我们随时可以用Variable.grad得到grad。
        # 刚创建Variable的时候，它的grad属性初始化为0.0
        loss.backward() # compute gradient and do SGD step
        optimizer.step() # 更新
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        output1 = torch.index_select(depth_pred.data, 1, torch.cuda.LongTensor([0]))
        result.evaluate(output1, target)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f}) '
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                  epoch, i+1, len(train_loader), data_time=data_time, 
                  gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3, 
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time})


def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda() # 从后面看，这里的target应该是深度图的ground truth
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        depth_pred = model(input_var)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        output1 = torch.index_select(depth_pred.data, 1, torch.cuda.LongTensor([0]))
        result.evaluate(output1, target)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50
        if args.modality == 'd':
            img_merge = None
        else:
            if args.modality == 'rgb':
                rgb = input
            elif args.modality == 'rgbd':
                rgb = input[:,:3,:,:]

            if i == 0:
                img_merge = utils.merge_into_row(rgb, target, depth_pred)
            # 隔50个图片抽一张作为可视化结果
            elif (i < 8*skip) and (i % skip == 0): # and等同于C++中的&&
                row = utils.merge_into_row(rgb, target, depth_pred)
                img_merge = utils.add_row(img_merge, row) # 添加一行
            elif i == 8*skip: # 只保存8张图片，保存够8张后输出
                filename = output_directory + '/comparison_' + str(epoch) + '.png' # str()：将()中的对象转换为字符串
                utils.save_image(img_merge, filename) # 建议：把这种常用的功能写到特定的脚本文件中，再像这样调用

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile: 
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3, 
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})

    return avg, img_merge

def save_checkpoint(state, is_best, epoch):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def adjust_learning_rate(optimizer, epoch):
    # """ """中的内容为函数的说明，在鼠标放在此函数上时会自动显示该说明
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    # //:整数除法，返回不大于结果的一个最大的整数
    #  /:浮点数除法。在符号前后的数字都为整型时，输出的也是整型，和c++一样
    # 在Python数学运算中*代表乘法，**为指数运算
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 这句话是程序的入口
# 意思是：当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
# 详见https://blog.csdn.net/yjk13703623757/article/details/77918633
if __name__ == '__main__':
    main()