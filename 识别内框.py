from maix import image, camera, display, app, nn, uart
import cv2
import sys
import struct
import math
import numpy as np

sys.path.append('/root/exam')
import serial_protocol

# 计算两点间距离（宽度）
def calc_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 计算比例系数k（宽度×距离）
def calc_k(width, distance):
    return width * distance

# 计算目标距离（k÷检测宽度）
def calc_target_distance(width, k):
    return k / width

# 初始化模型、摄像头、显示器
detector = nn.Retinaface(model="/root/models/retinaface.mud")
cam = camera.Camera(640,480)
disp = display.Display()

# 初始化串口通信（本版本不发送）
comm_proto = serial_protocol.SerialProtocol()
uart_device = "/dev/ttyS0"
serial_port = uart.UART(uart_device, 115200)

# 闭运算卷积核（用于图像预处理）
morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

while not app.need_exit():
    # 读取图像并转换格式
    img_frame = cam.read()
    img_bgr = image.image2cv(img_frame, copy=False)

    # 定义感兴趣区域 (ROI): 80% 正方形，居中
    height, width = img_bgr.shape[:2]  # 240x320
    roi_size = int(min(width, height) * 0.8)  # 192
    roi_x = (width - roi_size) // 2   # 64
    roi_y = (height - roi_size) // 2  # 24
    roi = img_bgr[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]

    # 图像预处理：灰度→闭运算→边缘检测（仅在 ROI 内）
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img=cv2.GaussianBlur(img,(5,5),0)
    #img = cv2.bilateralFilter(img_gray, 9, 30, 30)
    img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, morph_kernel)
    # img=cv2.GaussianBlur(img_closed,(5,5),0)

    img_edges = cv2.Canny(img, 50, 150)
    # img_show = image.cv2image(img_edges, copy=False) 
    # disp.show(img_show)
    # 查找所有轮廓（仅在 ROI 内）
    contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 筛选出符合条件的轮廓
        valid_contours = []
        
        for contour in contours:
            contour_area = cv2.contourArea(contour)

            if contour_area >= 500:
                # 多边形逼近
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx_corners = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx_corners) == 4:
                    # 排序角点（坐标调整回原图像）
                    corners = approx_corners.reshape(4, 2).astype(np.float32)
                    corners[:, 0] += roi_x  # x 偏移
                    corners[:, 1] += roi_y  # y 偏移
                    sum_vals = corners.sum(axis=1)
                    tl, br = corners[np.argmin(sum_vals)], corners[np.argmax(sum_vals)]
                    diff_vals = np.diff(corners, axis=1)
                    tr, bl = corners[np.argmin(diff_vals)], corners[np.argmax(diff_vals)]

                    # 计算各边长度
                    top_edge = calc_distance(tl[0], tl[1], tr[0], tr[1])
                    bottom_edge = calc_distance(bl[0], bl[1], br[0], br[1])
                    left_edge = calc_distance(tl[0], tl[1], bl[0], bl[1])
                    right_edge = calc_distance(tr[0], tr[1], br[0], br[1])
                    
                    # 判断矩形条件：上边≈下边，左边≈右边（误差不超过10像素）
                    top_bottom_diff = abs(top_edge - bottom_edge)
                    left_right_diff = abs(left_edge - right_edge)
                    
                    # 计算上边与侧边的比值
                    aspect_ratio1 = top_edge / left_edge
                    aspect_ratio2 = top_edge / right_edge
                    
                    # 判断是否满足条件：上下边相等，左右边相等（误差<10），上边/侧边≈0.7
                    is_valid_rectangle = (top_bottom_diff < 20 and 
                                         left_right_diff < 20 and
                                         abs(aspect_ratio1 - 0.7) < 0.15 and
                                         abs(aspect_ratio2 - 0.7) < 0.15)
                    
                    if is_valid_rectangle:
                        # 将符合条件的轮廓及其面积和角点信息保存
                        valid_contours.append({
                            'contour': contour,
                            'area': contour_area,
                            'corners': corners,
                            'tl': tl, 'tr': tr, 'bl': bl, 'br': br
                        })
        
        # 如果有符合条件的轮廓
        if valid_contours:
            # 按面积从小到大排序
            valid_contours.sort(key=lambda x: x['area'])
            
            # 选择面积最小的轮廓
            smallest_contour = valid_contours[0]
            
            # 提取角点信息
            corners = smallest_contour['corners']
            tl, tr, bl, br = smallest_contour['tl'], smallest_contour['tr'], smallest_contour['bl'], smallest_contour['br']
            left_edge = calc_distance(tl[0], tl[1], bl[0], bl[1])
            right_edge = calc_distance(tr[0], tr[1], br[0], br[1])
            # 计算侧边平均值
            side_edge_avg = (left_edge + right_edge) / 2
            print(f"侧边平均值: {side_edge_avg:.2f}px")
            # 中心点
            moments = cv2.moments(smallest_contour['contour'])
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"]) + roi_x
                center_y = int(moments["m01"] / moments["m00"]) + roi_y
                cv2.circle(img_bgr, (center_x, center_y), 5, (0, 0, 255), -1)

            # 计算宽度和距离
            detected_width = calc_distance(br[0], br[1], bl[0], bl[1])
            print(tl, tr, bl, br)
            print(br[0], br[1], bl[0], bl[1])
            k_val = calc_k(110, 15000)  
            target_dist = calc_target_distance(detected_width, k_val)

            print(f"检测宽度: {detected_width:.2f}px")
            print(f"比例系数k: {k_val}")
            print(f"目标距离: {target_dist:.2f}mm")
            print(f"上边/左边: {top_edge/left_edge:.2f}, 上边/右边: {top_edge/right_edge:.2f}")
            print(f"轮廓面积: {smallest_contour['area']:.2f}px²")

            # 绘制轮廓与距离
            cv2.drawContours(img_bgr, [corners.astype(np.int32).reshape(-1, 1, 2)], 0, (0, 255, 0), 2)
            cv2.putText(img_bgr, f" {target_dist:.1f}mm",
                        (center_x - 50, center_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 绘制角点编号
            corner_list = [tl, tr, br, bl]
            corner_labels = ["1", "2", "3", "4"]
            for i, (x, y) in enumerate(corner_list):
                cv2.circle(img_bgr, (int(x), int(y)), 5, (255, 0, 0), 2)
                cv2.putText(img_bgr, corner_labels[i],
                            (int(x) + 10, int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 显示 ROI 区域（绘制红色矩形边框）
    cv2.rectangle(img_bgr, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (0, 0, 255), 2)

    # 显示图像
    img_show = image.cv2image(img_bgr, copy=False)
    disp.show(img_show)
