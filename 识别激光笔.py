import cv2
import numpy as np

def is_circle_like(contour, threshold=0.7):
    area = cv2.contourArea(contour)
    ((x, y), radius) = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * (radius ** 2)
    return area / circle_area > threshold

cap = cv2.VideoCapture(1)  # 打开默认摄像头，如果有多个摄像头，可以尝试不同索引

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 分离通道
    b, g, r = cv2.split(frame)

    # 只保留红色通道，将绿色和蓝色通道设置为0
    zeros = np.zeros_like(r)
    red_only = cv2.merge([zeros, zeros, r])

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 找到灰度图的最大值
    max_value = np.max(gray)
    # 打印最大值
    print("灰度图像的最大值是:", max_value)

    # 应用阈值
    # 只保留红色通道的阈值
    # _, binary = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
    # 保留所有通道的阈值
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

     # 定义开运算核的大小和形状
    kernel = np.ones((10, 10), np.uint8)  # 可以根据需要调整核的大小和形状

    # 进行开运算操作
    binary_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(binary_opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (255, 255, 255), 2)

    # 绘制轮廓
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area and is_circle_like(cnt):
            # 绘制轮廓
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'G', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    image1 = cv2.resize(gray,(500,500))
    image2 = cv2.resize(binary_opened,(500,500))
    image3 = cv2.resize(frame,(500,500))
    horizontal_concat = cv2.hconcat([image1, image2])
    # 显示结果图像
    cv2.imshow('Detected', horizontal_concat)
    cv2.imshow('Laser', image3)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
