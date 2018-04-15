import cv2

rgb = cv2.imread('1.png')
gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
#gray = rgb[:,:,0] * 0.2989 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.114
orb = cv2.ORB_create(500)
kp = orb.detect(gray, None)
img2 = cv2.drawKeypoints(gray,kp,(225,0,0),-1)

cv2.namedWindow("rgb")
cv2.imshow("rgb", img2)
cv2.namedWindow("gray")
cv2.imshow("gray",gray)
cv2.waitKey(0)

    # def create_sparse_depth_ORB(self, rgb, depth, prob):
    #     num_samples = int(prob * depth.size)
    #     gray = rgb2grayscale(rgb)
    #     orb = cv2.ORB_create(num_samples)
    #     kp = orb.detect(gray, None)
    #     print len(kp)
    #     # img2 = cv2.drawKeypoints(gray, kp, (255,0,0), 1)

    #     mask_keep = np.random.uniform(0, 1, depth.shape) < prob # 生成一个0-1的mask，0-1是随机产生的，0-1产生的概率小于预设的概率
    #     sparse_depth = np.zeros(depth.shape)
    #     sparse_depth[mask_keep] = depth[mask_keep] # 把深度图中和mask_keep对应的元素赋值给sparse_depth中的对应元素
    #     return sparse_depth