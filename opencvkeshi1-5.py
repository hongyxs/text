import cv2
import numpy as np

# 读取图像
image_path = "E:\\work\\work\\text\\34_52.jpg"# 替换为实际的图像路径
image = cv2.imread(image_path)
print("图片数据模型：", image)    

# 检查图像是否成功读取
if image is None:
    print("错误：无法加载图像，请检查路径是否正确。")
    exit()
# 查看图片属性 
print("图片形状（高, 宽, 通道数）：", image.shape)   
print("图片数据类型：", image.dtype)    
print("图片像素总数：", image.size)

# 显示图像
cv2.imshow("Test Display Image", image)
print("按 's' 键保存原图，按 'k' 键进行处理，按 'e' 键边缘检测，按 'f' 键滤波，按 'm' 键形态学操作")

# 等待用户按键
key = cv2.waitKey(0)

# 保存图像功能
if key == ord('s'):  # 如果按下 's' 键
    # 修改文件名：添加前缀并改为PNG格式
    output_path = "E:\\work\\work\\text\\modified_34_52.png"
    success = cv2.imwrite(output_path, image)
    
    if success:
        print(f"图像已成功保存为 {output_path}")
        print(f"文件格式：PNG")
        print(f"保存状态：成功")
    else:
        print("图像保存失败，请检查文件路径和权限")

elif key == ord('k'):  # 如果按下 'k' 键
    print("开始图像处理流程：裁剪 → 缩放 → 水平翻转")
    
    # 裁剪为正方形 (826, 826)
    # 原图尺寸: 高=1223, 宽=826
    # 从底部开始裁剪，保留宽度不变
    start_y = 397  # 从顶部开始
    end_y = 1223  # 裁剪到826像素高度
    img_cropped = image[start_y:end_y, :]  # 高度裁剪，宽度不变

    # 缩放为 300×300
    img_resized = cv2.resize(img_cropped, (300, 300))

    # 水平翻转
    img_flipped = cv2.flip(img_resized, 1)  # 1表示水平翻转

    # 显示结果（可选）
    cv2.imshow("Cropped (826x826)", img_cropped)
    cv2.imshow("Resized (300x300)", img_resized)
    cv2.imshow("Flipped Horizontal", img_flipped)
    
    # 等待用户查看处理结果
    cv2.waitKey(0)
    
    # 保存最终结果
    output_path = "E:\\work\\work\\text\\processed_image.png"
    success = cv2.imwrite(output_path, img_flipped)

    if success:
        print(f"图像已成功保存为: {output_path}")
        print("处理流程: 裁剪 → 缩放 → 水平翻转")
    else:
        print("图像保存失败")

elif key == ord('e'):  # 如果按下 'e' 键
    print("开始边缘检测与轮廓分析")
    
    # 转换为灰度图
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化处理
    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Canny边缘检测
    edges = cv2.Canny(img_gray, 100, 200)
    
    # 提取轮廓（仅最外层轮廓）
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建轮廓分析图像
    img_contour = image.copy()
    
    print(f"检测到 {len(contours)} 个轮廓")
    
    # 绘制轮廓并分析特征
    for i, cnt in enumerate(contours):
        # 计算轮廓面积和周长
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)
        
        # 外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 绘制轮廓（绿色）
        cv2.drawContours(img_contour, [cnt], -1, (0, 255, 0), 2)
        
        # 绘制外接矩形（蓝色）
        cv2.rectangle(img_contour, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # 标注面积和周长信息（红色）
        cv2.putText(img_contour, f"Area:{int(area)}", (x, y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img_contour, f"Peri:{int(perimeter)}", (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        print(f"轮廓 {i+1}: 面积={int(area)}, 周长={int(perimeter)}")
    
    # 显示所有处理结果
    cv2.imshow("Gray Image", img_gray)
    cv2.imshow("Binary Image", img_bin)
    cv2.imshow("Canny Edges", edges)
    cv2.imshow("Contours Analysis", img_contour)
    
    # 等待用户查看
    cv2.waitKey(0)
    
    # 保存轮廓分析结果
    output_path = "E:\\work\\work\\text\\edge_analysis.png"
    success = cv2.imwrite(output_path, img_contour)
    
    if success:
        print(f"轮廓分析结果已保存为: {output_path}")
    else:
        print("轮廓分析结果保存失败")

elif key == ord('f'):  # 如果按下 'f' 键 - 图像滤波
    print("开始图像滤波处理")
    
    # 添加椒盐噪声（向量化版本）
    def add_salt_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
        noisy_image = image.copy()
        h, w = image.shape[0], image.shape[1]
        random_values = np.random.rand(h, w)
        
        salt_mask = random_values < salt_prob
        pepper_mask = (random_values >= salt_prob) & (random_values < salt_prob + pepper_prob)
        
        if len(image.shape) == 2:
            noisy_image[salt_mask] = 255
            noisy_image[pepper_mask] = 0
        else:
            noisy_image[salt_mask, :] = 255
            noisy_image[pepper_mask, :] = 0
        
        return noisy_image.astype(np.uint8)
    
    # 添加噪声
    img_noise = add_salt_pepper_noise(image)
    
    # 应用不同滤波方法
    img_blur = cv2.blur(img_noise, (5, 5))           # 均值滤波
    img_gaussian = cv2.GaussianBlur(img_noise, (5, 5), 1.5)  # 高斯滤波
    img_median = cv2.medianBlur(img_noise, 5)        # 中值滤波（对椒盐噪声效果好）
    img_bilateral = cv2.bilateralFilter(img_noise, 9, 75, 75)  # 双边滤波
    
    # 显示结果
    cv2.imshow("Original", image)
    cv2.imshow("Noise (Salt & Pepper)", img_noise)
    cv2.imshow("Mean Filter (5x5)", img_blur)
    cv2.imshow("Gaussian Filter (5x5)", img_gaussian)
    cv2.imshow("Median Filter (5x5)", img_median)
    cv2.imshow("Bilateral Filter", img_bilateral)
    
    print("滤波效果对比：")
    print("- 均值滤波：简单快速，但边缘模糊")
    print("- 高斯滤波：保留边缘较好，降噪效果佳")
    print("- 中值滤波：对椒盐噪声效果最好")
    print("- 双边滤波：保边降噪，但计算较慢")
    
    cv2.waitKey(0)
    
    # 保存滤波结果
    output_path = "E:\\work\\work\\text\\filter_comparison.png"
    # 创建对比图
    comparison = np.hstack([img_noise, img_median, img_bilateral])
    success = cv2.imwrite(output_path, comparison)
    
    if success:
        print(f"滤波对比图已保存为: {output_path}")
    else:
        print("滤波对比图保存失败")

elif key == ord('m'):  # 如果按下 'm' 键 - 形态学操作
    print("开始形态学操作")
    
    # 转换为灰度图
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 全局阈值二值化
    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    # 自适应阈值（对光照不均效果好）
    img_adaptive = cv2.adaptiveThreshold(img_gray, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    
    # 创建不同形状的结构元素
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    
    # 基本形态学操作
    img_erode = cv2.erode(img_bin, kernel_rect, iterations=1)    # 腐蚀
    img_dilate = cv2.dilate(img_bin, kernel_rect, iterations=1)  # 膨胀
    
    # 组合操作
    img_open = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_rect)   # 开运算（去噪点）
    img_close = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel_rect)  # 闭运算（补洞）
    img_gradient = cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, kernel_rect)  # 梯度运算
    
    # 显示结果
    cv2.imshow("Gray Image", img_gray)
    cv2.imshow("Binary (Global)", img_bin)
    cv2.imshow("Binary (Adaptive)", img_adaptive)
    cv2.imshow("Erosion", img_erode)
    cv2.imshow("Dilation", img_dilate)
    cv2.imshow("Opening (Remove Noise)", img_open)
    cv2.imshow("Closing (Fill Holes)", img_close)
    cv2.imshow("Gradient", img_gradient)
    
    print("形态学操作说明：")
    print("- 腐蚀：缩小白色区域，消除小物体")
    print("- 膨胀：扩大白色区域，填补空洞")
    print("- 开运算：先腐蚀后膨胀，去除噪点")
    print("- 闭运算：先膨胀后腐蚀，填补空洞")
    print("- 梯度：边缘检测，显示边界")
    
    cv2.waitKey(0)
    
    # 保存形态学结果
    output_path = "E:\\work\\work\\text\\morphology_operations.png"
    # 创建对比图
    comparison = np.hstack([img_bin, img_open, img_close])
    success = cv2.imwrite(output_path, comparison)
    
    if success:
        print(f"形态学操作对比图已保存为: {output_path}")
    else:
        print("形态学操作对比图保存失败")

else:  # 如果按下其他键
    print("图像未保存。")

# 关闭所有窗口
cv2.destroyAllWindows()
