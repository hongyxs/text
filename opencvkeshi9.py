import cv2
import numpy as np
import os

# 默认路径设置
DEFAULT_IMAGE_PATH = r"E:\work\work\text\34_52.jpg"#替换图片文件路径
DEFAULT_VIDEO_PATH = r"E:\work\work\text\Rick Astley - Never Gonna Give You Up.mp4"#替换视频文件路径

def detect_faces_in_image(image_path):
    """
    检测图片中的人脸
    """
    print(f"开始处理图片: {image_path}")
    
    # 1. 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误：图片文件不存在 {image_path}")
        return False
    
    # 2. 检查文件格式
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext not in valid_extensions:
        print(f"❌ 错误：不支持的图片格式 {file_ext}")
        print(f"支持的格式: {', '.join(valid_extensions)}")
        return False
    
    # 3. 加载人脸检测模型
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("❌ 错误：无法加载人脸分类器")
        return False
    
    print("✅ 人脸检测模型加载成功")
    
    # 4. 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 错误：无法读取图片 {image_path}")
        print("可能的原因：文件损坏或格式不支持")
        return False
    
    # 获取图片信息
    height, width = image.shape[:2]
    print(f"✅ 图片信息: {width}x{height} 像素")
    
    # 5. 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 6. 人脸检测
    print("正在进行人脸检测...")
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,    # 图像缩放比例
        minNeighbors=5,     # 检测敏感度
        minSize=(30, 30)    # 最小人脸尺寸
    )
    
    print(f"✅ 检测完成！共发现 {len(faces)} 张人脸")
    
    # 7. 绘制检测结果
    result_image = image.copy()
    for i, (x, y, w, h) in enumerate(faces):
        # 绘制绿色矩形框
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 添加人脸编号和置信度
        cv2.putText(result_image, f'Face {i+1}', (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 8. 添加信息面板
    info_text = f'Detected: {len(faces)} faces'
    cv2.putText(result_image, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(result_image, 'Press any key to close', (10, height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 9. 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Face Detection Result', result_image)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return True

def detect_faces_in_video(video_path):
    """
    检测视频文件中的人脸
    """
    print(f"开始处理视频: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ 错误：视频文件不存在 {video_path}")
        return False
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("❌ 无法加载人脸分类器")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频文件")
        return False
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ 视频信息: {width}x{height}, {fps:.1f} FPS")
    print("按 Q 退出，按 P 暂停")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'Q:Quit P:Pause', (10, 70),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Video Face Detection', frame)
        
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def realtime_camera_detection():
    """
    实时摄像头人脸检测
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("📷 实时摄像头检测中...")
    print("按 Q 退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Camera Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def check_files():
    """
    检查默认文件是否存在
    """
    print("🔍 检查默认文件...")
    
    image_exists = os.path.exists(DEFAULT_IMAGE_PATH)
    video_exists = os.path.exists(DEFAULT_VIDEO_PATH)
    
    print(f"默认图片: {'✅ 存在' if image_exists else '❌ 不存在'} - {DEFAULT_IMAGE_PATH}")
    print(f"默认视频: {'✅ 存在' if video_exists else '❌ 不存在'} - {DEFAULT_VIDEO_PATH}")
    
    return image_exists, video_exists

# 主程序
if __name__ == "__main__":
    print("=" * 65)
    print("🎭 OpenCV 全方位人脸检测系统")
    print("=" * 65)
    
    # 检查默认文件
    image_exists, video_exists = check_files()
    
    while True:
        print("\n请选择检测模式:")
        print("1. 图片人脸检测")
        print("2. 视频文件人脸检测")
        print("3. 实时摄像头人脸检测")
        print("4. 退出程序")
        
        choice = input("请输入选择 (1/2/3/4): ").strip()
        
        if choice == '1':
            if image_exists:
                detect_faces_in_image(DEFAULT_IMAGE_PATH)
            else:
                print("❌ 默认图片不存在")
                custom_path = input("请输入图片路径: ").strip()
                if custom_path:
                    detect_faces_in_image(custom_path)
                
        elif choice == '2':
            if video_exists:
                detect_faces_in_video(DEFAULT_VIDEO_PATH)
            else:
                print("❌ 默认视频不存在")
                custom_path = input("请输入视频路径: ").strip()
                if custom_path:
                    detect_faces_in_video(custom_path)
                
        elif choice == '3':
            print("启动实时摄像头检测...")
            realtime_camera_detection()
            
        elif choice == '4':
            print("🎶 程序退出")
            break
            
        else:
            print("❌ 无效选择")
