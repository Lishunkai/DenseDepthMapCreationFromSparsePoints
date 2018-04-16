import cv2
import numpy as np

rgb = cv2.imread('1.png')
gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
#gray = rgb[:,:,0] * 0.2989 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.114
orb = cv2.ORB_create(500)
kp = orb.detect(gray, None)
print(len(kp))

sparse_depth = np.zeros(rgb.shape)

for i in range(len(kp)):
    tu = kp[i].pt
    x = int(tu[0])
    y = int(tu[1])
    print(i,'  ',x,'  ',y)
    sparse_depth[y,x] = rgb[y,x] # x,y和行,列的顺序不一样


# print(len(kp)) # keypoint的个数
# print(kp[0]) # 多少个角点，就有多少个下标
# tu = kp[0].pt   #（提取坐标） pt指的是元组 tuple(x,y)
# print(tu[0],tu[1]) # 输出第一个keypoint的x,y坐标 


    # def create_sparse_depth_ORB(self, rgb, depth, prob):
    #     num_samples = int(prob * depth.size)
    #     gray = rgb2grayscale(rgb)
    #     orb = cv2.ORB_create(num_samples)
    #     kp = orb.detect(gray, None)
    #     print len(kp)

    #     mask_keep = np.random.uniform(0, 1, depth.shape) < prob # 生成一个0-1的mask，0-1是随机产生的，0-1产生的概率小于预设的概率
    #     sparse_depth = np.zeros(depth.shape)
    #     sparse_depth[mask_keep] = depth[mask_keep] # 把深度图中和mask_keep对应的元素赋值给sparse_depth中的对应元素
    #     return sparse_depth