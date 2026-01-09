# 导入 OpenCV 库
import cv2
import numpy as np

COLOR_RANGES = {
    # 红色（需要两个范围）
    'red_lower1': (0, 70, 50),
    'red_upper1': (10, 255, 255),
    'red_lower2': (170, 70, 50),
    'red_upper2': (180, 255, 255),
    
    # 橙色
    'orange_lower': (11, 100, 100),
    'orange_upper': (25, 255, 255),
    
    # 黄色
    'yellow_lower': (26, 100, 100),
    'yellow_upper': (34, 255, 255),
    
    # 绿色
    'green_lower': (35, 50, 50),
    'green_upper': (85, 255, 255),
    
    # 蓝色
    'blue_lower': (86, 50, 50),
    'blue_upper': (125, 255, 255),
    
    # 紫色
    'purple_lower': (126, 50, 50),
    'purple_upper': (150, 255, 255),
    
    # 粉色
    'pink_lower': (151, 30, 30),
    'pink_upper': (170, 255, 255),
    
    # 白色（需要低饱和度）
    'white_lower': (0, 0, 200),
    'white_upper': (180, 30, 255),
    
    # 黑色（需要低明度）
    'black_lower': (0, 0, 0),
    'black_upper': (180, 255, 50)
}


image_path = "E:\\work\\work\\text\\34_52.jpg"
image = cv2.imread(image_path)
print("图片数据模型：", image)    

# 检查图像是否成功读取
if image is None:
    print("错误：无法加载图像，请检查路径是否正确。")
    exit()
#转换为 HSV 颜色空间（HSV 对光照不敏感）
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义蓝色的范围，注意OpenCV中H的范围是[0,179]
# 蓝色在HSV中有两个范围：0-10和160-179
# 定义蓝色的 HSV 范围
lower_blue1 = np.array([86, 50, 50])
upper_blue1 = np.array([125, 255, 255])

# 生成掩码（只保留蓝色区域）
mask = cv2.inRange(hsv, lower_blue1, upper_blue1)

# 提取红色区域
result = cv2.bitwise_and(image, image, mask=mask)

# 分离HSV通道
h, s, v = cv2.split(hsv)

#显示HSV通道
cv2.imshow('Hue', h)        # 色相通道，显示为灰度图，因为值代表色相
cv2.imshow('Saturation', s) # 饱和度通道
cv2.imshow('Value', v)      # 明度通道
cv2.waitKey(0)
cv2.destroyAllWindows()

# 显示
cv2.imshow("Original", image)
cv2.imshow("Mask", mask)
cv2.imshow("blue Objects", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#课后作业：修改代码检测蓝色物体，调整 HSV 阈值。

