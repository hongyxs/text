import cv2
import numpy as np

def detect_camera_color(target_color='red'):
    """
    实时检测摄像头画面中的特定颜色
    
    Args:
        target_color: 要检测的颜色 ('red', 'blue', 'green', 'yellow')
    """
    # 定义颜色范围
    color_ranges = {
        'red': [
            (np.array([0, 120, 70]), np.array([10, 255, 255])),  # 红色范围1
            (np.array([170, 120, 70]), np.array([180, 255, 255])) # 红色范围2
        ],
        'blue': [(np.array([100, 150, 50]), np.array([130, 255, 255]))],
        'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
        'yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))]
    }
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        return
    
    print(f"开始实时检测 {target_color} 颜色，按 'q' 键退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 创建颜色掩码
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        if target_color in color_ranges:
            for lower, upper in color_ranges[target_color]:
                mask += cv2.inRange(hsv, lower, upper)
        
        # 形态学操作去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 在帧上绘制检测结果
        display_frame = frame.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        object_count = 0
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                object_count += 1
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, f'{target_color}', 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示统计信息
        cv2.putText(display_frame, f'{target_color} Objects: {object_count}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 显示检测结果
        cv2.imshow('Color Detection', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    # 实时检测摄像头中的红色物体
    detect_camera_color('red')



