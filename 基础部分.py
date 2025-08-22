from maix import image, camera, display, app, nn, uart
import cv2
import sys
import struct
import math
import numpy as np
import collections
sys.path.append('/root/exam')
import serial_protocol
comm_proto = serial_protocol.SerialProtocol()
device = "/dev/ttyS0"
serial = uart.UART(device, 115200)
def is_equilateral_triangle(points):
    # Calculate side lengths
    side_1 = math.dist(points[0], points[1])
    side_2 = math.dist(points[1], points[2])
    side_3 = math.dist(points[2], points[0])
    
    # Define a tolerance for side length differences
    tolerance = 5  # Adjust tolerance as needed

    # # Check if all sides are approximately the same length
    # if abs(side_1 - side_2) < tolerance and abs(side_2 - side_3) < tolerance:
    #     average_side_length = (side_1 + side_2 + side_3) / 3
    #     return True, average_side_length  # 返回平均边长
    # return False, 0  # 统一返回元组，避免解包错误
    if abs(side_1 - side_2) < tolerance and abs(side_2 - side_3) < tolerance:
        max_side_length = max(side_1, side_2, side_3)  # 计算最大边长
        return True, max_side_length  # 返回最大边长
    return False, 0  # 统一返回元组，避免解包错误  

# 优化顶点排序函数，确保按顺时针或逆时针顺序排列
def sort_triangle_points(points):
    # 计算质心
    centroid = np.mean(points, axis=0)
    
    # 计算每个点相对于质心的角度
    angles = []
    for point in points:
        dx = point[0] - centroid[0]
        dy = point[1] - centroid[1]
        angle = math.atan2(dy, dx)
        angles.append(angle)
    
    # 根据角度排序点，实现顺时针或逆时针排序
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    # 确定顶部顶点（y值最小的点）
    top_index = np.argmin(sorted_points[:, 1])
    
    # 重新排列，使顶部顶点成为第一个点
    reordered_points = np.roll(sorted_points, -top_index, axis=0)
    
    return reordered_points

def is_square(approx, side_tolerance=0.2, angle_tolerance=10, aspect_ratio_tolerance=0.15):
    """
    严格判断四边形是否为正方形
    side_tolerance: 边长差异容忍度（0-1之间，默认10%）
    angle_tolerance: 角度容忍度（度数，默认±10度）
    aspect_ratio_tolerance: 长宽比容忍度（默认15%）
    """
    if len(approx) != 4:
        return False
    
    # 获取四个顶点
    points = approx.reshape(-1, 2).astype(np.float32)
    
    # 方法1: 检查边长是否相等
    sides = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        side_length = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        sides.append(side_length)
    
    # 检查四条边长度是否相近
    avg_side = np.mean(sides)
    if avg_side < 10:  # 避免太小的形状
        return False
        
    for side in sides:
        if abs(side - avg_side) / avg_side > side_tolerance:
            print(f"边长不符合: {sides}, 平均: {avg_side:.2f}")
            return False
    
    # 方法2: 检查外接矩形的长宽比
    rect = cv2.minAreaRect(approx)
    width, height = rect[1]
    if width == 0 or height == 0:
        return False
        
    aspect_ratio = max(width, height) / min(width, height)
    if abs(aspect_ratio - 1.0) > aspect_ratio_tolerance:
        print(f"长宽比不符合: {aspect_ratio:.2f}")
        return False
    
    # 方法3: 检查角度是否接近90度
    angles = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        p3 = points[(i + 2) % 4]
        
        # 计算向量
        v1 = p1 - p2
        v2 = p3 - p2
        
        # 避免零向量
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            continue
            
        # 计算角度
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle) * 180 / np.pi
        angles.append(angle)
    
    if len(angles) < 4:
        return False
        
    for angle in angles:
        if abs(angle - 90) > angle_tolerance:
            print(f"角度不符合: {angles}")
            return False
    
    # 方法4: 检查对角线长度是否相等
    diag1 = np.sqrt((points[0][0] - points[2][0])**2 + (points[0][1] - points[2][1])**2)
    diag2 = np.sqrt((points[1][0] - points[3][0])**2 + (points[1][1] - points[3][1])**2)
    
    if abs(diag1 - diag2) / max(diag1, diag2) > side_tolerance:
        print(f"对角线不相等: {diag1:.2f}, {diag2:.2f}")
        return False
    
    print(f"正方形检测通过: 边长={sides}, 角度={angles}, 长宽比={aspect_ratio:.2f}")
    return True, avg_side  # 返回平均边长

# 计算两点间距离（宽度）
def calc_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 计算比例系数k（宽度×距离）
def calc_k(width, distance):
    return width * distance

# 计算目标距离（k÷检测宽度）
def calc_target_distance(width, k):
    return k / width

# 计算正方形边长对应的实际尺寸
def calc_real_size(pixel_size, distance_mm):
    """
    根据正方形的像素边长和距离，计算实际边长
    pixel_size: 像素边长
    distance_mm: 距离（毫米）
    返回: 实际边长（毫米）
    """
    # 使用相机标定参数或经验公式
    # 这里使用简化的线性关系：实际尺寸 = 像素尺寸 * 距离系数
    # 距离系数需要根据实际相机参数调整
    distance_factor = distance_mm / 1000.0  # 转换为米
    # 假设在1米距离时，1个像素约等于1mm（需要根据实际情况校准）
    real_size = pixel_size * distance_factor * 0.5  # 0.5是经验系数，需要校准
    return real_size

# 初始化模型、摄像头、显示器
detector = nn.Retinaface(model="/root/models/retinaface.mud")
cam = camera.Camera(320, 240)
disp = display.Display()
detected_width=1
# 初始化串口通信（本版本不发送）
comm_proto = serial_protocol.SerialProtocol()
uart_device = "/dev/ttyS0"
serial_port = uart.UART(uart_device, 115200)

# 闭运算卷积核（用于图像预处理）
morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 已知参考物体的参数（用于距离计算）
REFERENCE_WIDTH_MM = 68  # 参考物体实际宽度（毫米）
REFERENCE_DISTANCE_MM = 15000  # 参考距离（毫米）
K_VALUE = calc_k(REFERENCE_WIDTH_MM, REFERENCE_DISTANCE_MM)  # 比例系数
# 在类或函数外部初始化
distance_buffer = collections.deque(maxlen=10)
confirmed_distance = None
is_distance_confirmed = False
size_history = collections.deque(maxlen=10)
stable_size = None
size_locked = False
triangle_size_history = collections.deque(maxlen=10)
stable_triangle_size = None
triangle_size_locked = False
circle_size_history = collections.deque(maxlen=10)
stable_circle_size = None
circle_size_locked = False
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
    img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img_closed = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, morph_kernel)
    #img = cv2.GaussianBlur(img_closed, (5, 5), 0)
    img = cv2.bilateralFilter(img_closed, 9, 30, 30)
    img_edges = cv2.Canny(img, 50, 150)
    
    # 查找所有轮廓（仅在 ROI 内）
    contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 处理最大轮廓（原有的距离测量逻辑）
        max_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(max_contour)

        if contour_area >= 1500:
            # 多边形逼近
            epsilon = 0.02 * cv2.arcLength(max_contour, True)
            approx_corners = cv2.approxPolyDP(max_contour, epsilon, True)

            if len(approx_corners) == 4:
                # 排序角点（坐标调整回原图像）
                corners = approx_corners.reshape(4, 2).astype(np.float32)
                corners[:, 0] += roi_x  # x 偏移
                corners[:, 1] += roi_y  # y 偏移
                sum_vals = corners.sum(axis=1)
                tl, br = corners[np.argmin(sum_vals)], corners[np.argmax(sum_vals)]
                diff_vals = np.diff(corners, axis=1)
                tr, bl = corners[np.argmin(diff_vals)], corners[np.argmax(diff_vals)]

                # 中心点
                moments = cv2.moments(approx_corners)
                if moments["m00"] != 0:
                    center_x = int(moments["m10"] / moments["m00"]) + roi_x
                    center_y = int(moments["m01"] / moments["m00"]) + roi_y
                    cv2.circle(img_bgr, (center_x, center_y), 5, (0, 0, 255), -1)

                # 计算宽度和距离
                detected_width = calc_distance(br[0], br[1], bl[0], bl[1])
                target_dist = calc_target_distance(detected_width, K_VALUE)
                target_dist=target_dist/100
                print(f"检测宽度: {detected_width:.2f}px")
                print(f"目标距离: {target_dist:.2f}mm")
                # 添加当前距离到缓冲区
                distance_buffer.append(target_dist)
                print('2',distance_buffer)
                if not is_distance_confirmed:
                    # 检查是否有10帧数据且都稳定
                    if len(distance_buffer) == 10:
                        distances = list(distance_buffer)
                        max_diff = max(distances) - min(distances)

                        if max_diff <= 5.0:  # 0.5cm = 5mm
                            confirmed_distance = sum(distances) / len(distances)
                            is_distance_confirmed = True
                            print(f"距离确认: {confirmed_distance:.2f}mm")
                            serial.write(f"D:({confirmed_distance:.2f})".encode('utf-8'))
                else:
                    # 检查距离变动是否超过1cm
                    if abs(target_dist - confirmed_distance) > 10.0:  # 1cm = 10mm
                        print("距离变动超过1cm,重新检测")
                        is_distance_confirmed = False
                        confirmed_distance = None
                        distance_buffer.clear()

                # serial.write(f"Error:({dx_red},{dy_red})".encode('utf-8'))
                # 绘制轮廓与距离
                cv2.drawContours(img_bgr, [corners.astype(np.int32).reshape(-1, 1, 2)], 0, (0, 255, 0), 2)
                cv2.putText(img_bgr, f" {target_dist:.1f}mm",
                            (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # 绘制角点编号
                corner_list = [tl, tr, br, bl]
                corner_labels = ["1", "2", "3", "4"]
                for i, (x, y) in enumerate(corner_list):
                    cv2.circle(img_bgr, (int(x), int(y)), 5, (255, 0, 0), 2)
                    cv2.putText(img_bgr, corner_labels[i],
                                (int(x) + 10, int(y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 处理所有轮廓，寻找正方形
        for contour in contours: 
            # 圆度阈值
            min_circularity = 0.85
            # 过滤掉太小的轮廓
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 50 or area < 50:
                continue
            if area < 100:  # 提高最小面积阈值
                continue
                
            # 计算参数，多边形逼近
            epsilon = 0.03 * cv2.arcLength(contour, True) 
            approx = cv2.approxPolyDP(contour, epsilon, True)  
            
            # 只处理四边形
            if len(approx) == 4:


                # 修正is_square函数调用参数（原代码传入了错误的坐标参数）
                square_result = is_square(approx)
                if isinstance(square_result, tuple) and square_result[0]:
                    corners = approx.reshape(4, 2).astype(np.float32)
                    corners[:, 0] += roi_x  # x 偏移
                    corners[:, 1] += roi_y  # y 偏移
                    sum_vals = corners.sum(axis=1)
                    tl, br = corners[np.argmin(sum_vals)], corners[np.argmax(sum_vals)]
                    diff_vals = np.diff(corners, axis=1)
                    tr, bl = corners[np.argmin(diff_vals)], corners[np.argmax(diff_vals)]
                    # 正方形处理
                    is_square_flag, avg_side_length = square_result
                    print()
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"]) + roi_x  # 添加ROI的x偏移
                        cy = int(M["m01"] / M["m00"]) + roi_y  # 添加ROI的y偏移
                        cv2.circle(img_bgr, (cx, cy), 5, (0, 0, 255), -1)
                        # 计算正方形的实际距离（使用边长）
                        square_distance = calc_target_distance(avg_side_length, K_VALUE)
                        
                        # 计算正方形的实际尺寸
                        real_size = ((avg_side_length / detected_width) * 22)
                        # 添加当前尺寸到历史记录
                        size_history.append(real_size)

                        # 尺寸稳定性检测
                        if not size_locked:
                            if len(size_history) == 10:
                                size_list = list(size_history)
                                size_range = max(size_list) - min(size_list)

                                if size_range <= 0.5:  
                                    stable_size = sum(size_list) / len(size_list)
                                    size_locked = True
                                    print(f"尺寸锁定: {stable_size:.2f}mm")
                                    serial.write(f"X:({stable_size:.2f})".encode('utf-8'))
                        else:
                            if abs(real_size - stable_size) > 10.0:  # 1cm = 10mm
                                print("尺寸变动超过1cm，重新检测")
                                size_locked = False
                                stable_size = None
                                size_history.clear()                       
                        cv2.putText(img_bgr, f"Real: {real_size:.1f}mm", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    # 发送数据
                    offset_approx = approx.copy()  # 复制轮廓用于坐标偏移
                    offset_approx[:, :, 0] += roi_x  # x坐标偏移
                    offset_approx[:, :, 1] += roi_y  # y坐标偏移
                    points = offset_approx.reshape(-1, 2)
                    
                    # print("="*50)
                    # print("Square detected:")
                    # print(f"  像素边长1: {detected_width:.2f} px")
                    # print(f"  像素边长: {avg_side_length:.2f} px")
                    # print(f"  估算距离: {square_distance:.2f} mm")
                    print(f"  实际尺寸: {real_size:.2f} mm")
                    # print(f"  顶点坐标: {points}")
                    # print("="*50)
                    
                    # 绿色粗线标注正方形
                    cv2.drawContours(img_bgr, [offset_approx], 0, (0, 255, 0), 3) 
                    
                    # 标注顶点
                    for i, point in enumerate(offset_approx):
                        x, y = point.ravel()
                        cv2.circle(img_bgr, (x, y), 8, (255, 0, 0), 2)
                        cv2.putText(img_bgr, str(i+1), (x-5, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                elif isinstance(square_result, bool) and square_result:
                    # 兼容原来的返回格式
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"]) + roi_x
                        cy = int(M["m01"] / M["m00"]) + roi_y
                        cv2.circle(img_bgr, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(img_bgr, "SQUARE", (cx-30, cy-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    offset_approx = approx.copy()
                    offset_approx[:, :, 0] += roi_x
                    offset_approx[:, :, 1] += roi_y
                    points = offset_approx.reshape(-1, 2)
                    print("Square detected:", points)
                    
                    cv2.drawContours(img_bgr, [offset_approx], 0, (0, 255, 0), 3) 
                    
                    for point in offset_approx:
                        x, y = point.ravel()
                        cv2.circle(img_bgr, (x, y), 8, (255, 0, 0), 2)
            if len(approx) == 3:  # Check for triangles
                points = approx.reshape(-1, 2)  # Convert to (n, 2) array
                
                # 只调用一次函数
                is_equilateral, average_side_length = is_equilateral_triangle(points)
                
                if is_equilateral and detected_width!=0:  # 如果是等边三角形
                    print(f"等边三角形边长: {average_side_length:.2f} 像素")
                    
                    # 优化顶点排序
                    sorted_points = sort_triangle_points(points)
                    
                    # 顶点标签（1, 2, 3）
                    labels = ["1", "2", "3"]
                    
                    # 计算中心点（坐标调整回原图像）
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"]) + roi_x  # 添加ROI的x偏移
                        cy = int(M["m01"] / M["m00"]) + roi_y  # 添加ROI的y偏移
                        cv2.circle(img_bgr, (cx, cy), 5, (0, 0, 255), -1)
                        
                        # 计算三角形的实际距离（使用边长）
                        triangle_distance = calc_target_distance(average_side_length, K_VALUE)
                        
                        # 计算三角形的实际尺寸（使用与正方形相同的比例关系）
                        triangle_real_size = ((average_side_length / detected_width) * 23)
                        # 添加当前三角形尺寸到历史记录
                        triangle_size_history.append(triangle_real_size)

                        # 三角形尺寸稳定性检测
                        if not triangle_size_locked:
                            if len(triangle_size_history) == 10:
                                triangle_size_list = list(triangle_size_history)
                                triangle_size_range = max(triangle_size_list) - min(triangle_size_list)

                                if triangle_size_range <= 5.0:  # 0.5cm = 5mm
                                    stable_triangle_size = sum(triangle_size_list) / len(triangle_size_list)
                                    triangle_size_locked = True
                                    print(f"三角形尺寸锁定: {stable_triangle_size:.2f}mm")
                                    serial.write(f"X:({stable_triangle_size:.2f})".encode('utf-8'))
                        else:
                            if abs(triangle_real_size - stable_triangle_size) > 10.0:  # 1cm = 10mm
                                print("三角形尺寸变动超过1cm,重新检测")
                                triangle_size_locked = False
                                stable_triangle_size = None
                                triangle_size_history.clear()
                        # 显示实际尺寸
                        cv2.putText(img_bgr, f"Real: {triangle_real_size:.1f}mm", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    # Draw the triangle and highlight the points
                    offset_approx = approx.copy()  # 复制轮廓用于坐标偏移
                    offset_approx[:, :, 0] += roi_x  # x坐标偏移
                    offset_approx[:, :, 1] += roi_y  # y坐标偏移
                    cv2.drawContours(img_bgr, [offset_approx], 0, (0, 255, 0), 2)
                    
                    # 标记所有顶点并添加标签
                    for i, point in enumerate(sorted_points):
                        x, y = point
                        x += roi_x  # 添加ROI的x偏移
                        y += roi_y  # 添加ROI的y偏移
                        color = (0, 0, 255) if i == 0 else (255, 0, 0)  # 第一个点为红色，其余为蓝色
                        cv2.circle(img_bgr, (int(x), int(y)), 8, color, -1)
                        cv2.putText(img_bgr, labels[i], (int(x)+10, int(y)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # # 显示三角形信息
                    # cv2.putText(img_bgr, f"边长: {average_side_length:.1f}px", 
                    #         (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) 
                    
                    # print("="*50)
                    # print("Equilateral Triangle detected:")
                    # print(f"  三角形像素边长: {average_side_length:.2f} px")
                    # print(f"  估算距离: {triangle_distance:.2f} mm")
                    # print(f"  实际尺寸: {real_size:.2f} mm")
                    # print(f"  顶点坐标: {offset_approx.reshape(-1, 2)}")
                    # print("="*50)
            # 计算圆度(该指标是度量区域形状接近圆形的指标，值越接近1，形状越接近圆形)
            circularity = 4 * np.pi * area / (perimeter**2)
            if circularity > min_circularity:
                # 最优椭圆拟合(内部采用最小二乘法进行拟合计算)
                center, axes, angle = cv2.fitEllipse(contour)
                
                # 添加ROI偏移量
                center = (center[0] + roi_x, center[1] + roi_y)
                major_axis = max(axes)  # 长轴作为直径
                minor_axis = min(axes)  # 短轴
                
                # 计算圆的半径（取长短轴平均值）
                radius = (major_axis + minor_axis) / 4  # 除以2得到半径，再取平均
                
                # 计算圆的实际距离
                circle_distance = calc_target_distance(major_axis, K_VALUE)
                
                # 计算圆的实际尺寸（直径）
                circle_real_size = ((major_axis / detected_width) * 21.5) if detected_width != 0 else 0
                # 添加当前圆形尺寸到历史记录
                circle_size_history.append(circle_real_size)

                # 圆形尺寸稳定性检测
                if not circle_size_locked:
                    if len(circle_size_history) == 10:
                        circle_size_list = list(circle_size_history)
                        circle_size_range = max(circle_size_list) - min(circle_size_list)

                        if circle_size_range <= 5.0:  # 0.5cm = 5mm
                            stable_circle_size = sum(circle_size_list) / len(circle_size_list)
                            circle_size_locked = True
                            print(f"圆形尺寸锁定: {stable_circle_size:.2f}mm")
                            serial.write(f"X:({stable_circle_size:.2f})".encode('utf-8'))
                else:
                    if abs(circle_real_size - stable_circle_size) > 10.0:  # 1cm = 10mm
                        print("圆形尺寸变动超过1cm,重新检测")
                        circle_size_locked = False
                        stable_circle_size = None
                        circle_size_history.clear()
                # 画椭圆并标识圆心
                cv2.ellipse(img_bgr, (center, axes, angle), (0, 255, 0), 1)
                cv2.drawMarker(img_bgr, 
                  (int(center[0]), int(center[1])), 
                  (255, 0, 0),  # 蓝色(BGR)
                  markerType=cv2.MARKER_CROSS, 
                  markerSize=10, 
                  thickness=2)
                
                # 显示圆的信息
                # cv2.putText(img_bgr, f"Radius: {radius:.1f}px", 
                #           (int(center[0]) + 20, int(center[1]) - 20),
                #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(img_bgr, f"Real: {circle_real_size:.1f}mm", 
                          (20,40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # print("="*50)
                # print("Circle detected:")
                # print(f"  圆心坐标: {center}")
                # print(f"  长轴: {major_axis:.2f} px, 短轴: {minor_axis:.2f} px")
                # print(f"  估算距离: {circle_distance:.2f} mm")
                # print(f"  实际尺寸: {real_size:.2f} mm")
                # print("="*50)
    # 显示 ROI 区域（绘制红色矩形边框）
    cv2.rectangle(img_bgr, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (0, 0, 255), 2)

    # 显示图像
    img_show = image.cv2image(img_bgr, copy=False)
    disp.show(img_show)