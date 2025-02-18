import os
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import re
import json
import csv

def lines_intersect(line1_start, line1_end, line2_start, line2_end):
    """检查两条线段是否相交"""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    A = line1_start
    B = line1_end
    C = line2_start
    D = line2_end
    
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def box_lines_intersect(box1, box2):
    """检查两个框的边线是否相交"""
    x1_1, y1_1, x2_1, y2_1 = map(int, box1)
    x1_2, y1_2, x2_2, y2_2 = map(int, box2)
    
    # 第一个框的四条边
    box1_lines = [
        ((x1_1, y1_1), (x2_1, y1_1)),  # 上边
        ((x2_1, y1_1), (x2_1, y2_1)),  # 右边
        ((x2_1, y2_1), (x1_1, y2_1)),  # 下边
        ((x1_1, y2_1), (x1_1, y1_1))   # 左边
    ]
    
    # 第二个框的四条边
    box2_lines = [
        ((x1_2, y1_2), (x2_2, y1_2)),  # 上边
        ((x2_2, y1_2), (x2_2, y2_2)),  # 右边
        ((x2_2, y2_2), (x1_2, y2_2)),  # 下边
        ((x1_2, y2_2), (x1_2, y1_2))   # 左边
    ]
    
    # 检查任意两条边是否相交
    for line1 in box1_lines:
        for line2 in box2_lines:
            if lines_intersect(line1[0], line1[1], line2[0], line2[1]):
                return True
    
    return False

def boxes_intersect(box1, box2):
    """检查两个框是否有显著重叠"""
    x1_1, y1_1, x2_1, y2_1 = map(int, box1)
    x1_2, y1_2, x2_2, y2_2 = map(int, box2)
    
    # 计算重叠区域
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    # 如果没有重叠，直接返回False
    if x_right < x_left or y_bottom < y_top:
        return False
        
    # 计算重叠面积
    overlap_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算两个框的面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 只有当重叠面积超过较小框面积的30%时才认为是重叠
    smaller_area = min(area1, area2)
    return overlap_area > 0.3 * smaller_area

def filter_boxes(boxes, scores):
    """过滤所有边相交的框，只保留置信度最高的"""
    if len(boxes) == 0:
        return [], []
    
    # 根据置信度排序
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    scores = scores[indices]
    
    keep = []
    
    # 遍历所有框
    for i in range(len(boxes)):
        keep_box = True
        # 检查是否与任何已保留的框相交
        for j in range(len(keep)):
            if boxes_intersect(boxes[i], boxes[keep[j]]):
                keep_box = False
                break
        if keep_box:
            keep.append(i)
    
    return boxes[keep], scores[keep]

def extract_coordinates(filename):
    """从文件名中提取x和y坐标"""
    pattern = r'x(\d+)_y(\d+)'
    match = re.search(pattern, filename)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return x, y
    return None, None

# 配置参数
PIXEL_RESOLUTION = 100.0000418  # 原始像素分辨率（米/像素）
SCALE_FACTOR = 1.0  # 缩放因子，例如：0.5表示缩小一半，2表示放大一倍

# 根据缩放计算新的像素分辨率
ADJUSTED_RESOLUTION = PIXEL_RESOLUTION / SCALE_FACTOR

# 加载YOLOv8模型
model = YOLO('yolov8_ADown.pt')

# 存储结果的列表
results_data = []

def check_boxes_overlap(box1, box2):
    """检查两个框是否相交（简单的坐标比较）"""
    x1_1, y1_1, x2_1, y2_1 = map(int, box1)
    x1_2, y1_2, x2_2, y2_2 = map(int, box2)
    
    # 如果一个框在另一个框的完全左边或右边，则不相交
    if x2_1 < x1_2 or x1_1 > x2_2:
        return False
    # 如果一个框在另一个框的完全上边或下边，则不相交
    if y2_1 < y1_2 or y1_1 > y2_2:
        return False
    return True

def process_small_images(model_path, small_images_folder, ref_image_path, boxes_json_path, params):
    """处理小图片和参考图，保存框信息到json"""
    # 确保输出目录存在
    output_dir = os.path.dirname(boxes_json_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    model = YOLO(model_path)
    first_pass_boxes = []
    
    # 分别设置两次识别的置信度阈值
    first_pass_conf = 0.4  # 第一次识别（小图）的阈值
    second_pass_conf = 0.15  # 第二次识别（大图）的阈值
    print(f"Using confidence thresholds: first_pass={first_pass_conf}, second_pass={second_pass_conf}")
    
    # 获取参考图尺寸
    ref_img = cv2.imread(ref_image_path)
    if ref_img is None:
        raise ValueError(f"Failed to read reference image: {ref_image_path}")
    ref_height, ref_width = ref_img.shape[:2]
    
    # 第一次识别：处理小图
    print("\nFirst pass: detecting on small images...")
    all_boxes = []
    all_scores = []
    
    for filename in sorted([f for f in os.listdir(small_images_folder) if f.endswith('.tif')]):
        try:
            img_path = os.path.join(small_images_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # 从文件名解析位置信息
            parts = filename.split('_')
            if len(parts) >= 9:
                row = int(parts[6][1:])
                col = int(parts[7][1:])
                
                # 计算在大图中的位置
                x_start = col * (ref_width // 8)
                y_start = row * (ref_height // 8)
            else:
                continue
            
            # 使用第一次识别的阈值
            results = model.predict(img, conf=first_pass_conf)
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # 收集所有框和置信度
                for box, conf in zip(boxes, confidences):
                    adjusted_box = [
                        float(box[0]) + x_start,
                        float(box[1]) + y_start,
                        float(box[2]) + x_start,
                        float(box[3]) + y_start
                    ]
                    all_boxes.append(adjusted_box)
                    all_scores.append(conf)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # 使用NMS处理同一次检测内的重复框
    if all_boxes:
        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        first_pass_boxes, _ = filter_boxes(all_boxes, all_scores)
        # 将numpy数组转换为普通Python列表
        first_pass_boxes = first_pass_boxes.tolist()
    else:
        first_pass_boxes = []
    
    print(f"\nFirst pass detected {len(first_pass_boxes)} boxes")
    
    # 第二次识别：使用较低的阈值
    print("\nSecond pass: detecting on reference image...")
    results = model.predict(ref_img, conf=second_pass_conf)
    second_pass_boxes = []
    
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confidences):
            # 直接转换为列表
            second_pass_boxes.append(box.tolist())
            print(f"Found box with confidence {conf:.2f}: {box.tolist()}")
    
    print(f"Second pass detected {len(second_pass_boxes)} boxes")
    
    # 保存框信息
    boxes_data = {
        'first_pass': first_pass_boxes,  # 现在是普通Python列表
        'second_pass': second_pass_boxes,  # 现在是普通Python列表
        'thresholds': {
            'first_pass': first_pass_conf,
            'second_pass': second_pass_conf
        }
    }
    
    with open(boxes_json_path, 'w') as f:
        json.dump(boxes_data, f)
    print(f"Saved boxes to {boxes_json_path}")
    
    # 保存final_boxes.json (合并所有框)
    final_boxes_path = os.path.join(os.path.dirname(boxes_json_path), 'final_boxes.json')
    with open(final_boxes_path, 'w') as f:
        json.dump({'boxes': first_pass_boxes + second_pass_boxes}, f)
    print(f"Saved final boxes to {final_boxes_path}")
    
    return boxes_data

def generate_csv_from_json(boxes_json_path, csv_path, params):
    """从json生成csv报告，包含直径计算"""
    try:
        # 原始比例是100.0000418 m/pixel
        # GAN处理后图像边长翻倍，所以新的比例是50.0000209 m/pixel
        SCALE_FACTOR = 50.0000209  # m/pixel
        
        with open(boxes_json_path, 'r') as f:
            data = json.load(f)
            boxes = data.get('boxes', [])
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'X1', 'Y1', 'X2', 'Y2', 'Width', 'Height', 
                           'Diameter_Pixels', 'Diameter_KM'])
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(float, box)
                width = abs(x2 - x1)   # 使用绝对值
                height = abs(y2 - y1)  # 使用绝对值
                
                # 取宽度和高度的最大值作为直径（像素）
                diameter_pixels = max(width, height)
                
                # 转换为实际距离（千米）
                # 将米转换为千米需要除以1000
                diameter_km = (diameter_pixels * SCALE_FACTOR) / 1000
                
                writer.writerow([
                    i+1,           # ID
                    x1, y1, x2, y2,  # 坐标
                    width, height,    # 宽度和高度（像素）
                    diameter_pixels,  # 直径（像素）
                    f"{diameter_km:.3f}"  # 直径（千米，保留3位小数）
                ])
        
        print(f"Generated CSV report with diameters: {csv_path}")
        
    except Exception as e:
        print(f"Error generating CSV report: {str(e)}")
        import traceback
        traceback.print_exc()

def process_large_image(model_path, merged_image_path, boxes_json_path, green_boxes_path, params):
    """处理大图，只输出绿框"""
    model = YOLO(model_path)
    
    # 读取大图
    merged_img = cv2.imread(merged_image_path)
    if merged_img is None:
        raise ValueError("Failed to read merged image")
    
    # 对大图进行检测
    results = model.predict(merged_img, conf=params['conf_threshold'])
    green_boxes = []
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        # 直接保存所有检测到的框，不做尺寸限制
        green_boxes = boxes.tolist()
    
    # 保存绿框信息
    with open(green_boxes_path, 'w') as f:
        json.dump({'green_boxes': green_boxes}, f)

def process_and_draw_boxes(ref_image_path, boxes_json_path, final_boxes_path):
    """在参考图上绘制所有检测到的框"""
    ref_img = cv2.imread(ref_image_path)
    if ref_img is None:
        raise ValueError(f"Failed to read reference image: {ref_image_path}")
    
    output_img = ref_img.copy()
    
    # 读取框信息
    with open(boxes_json_path, 'r') as f:
        data = json.load(f)
        first_pass_boxes = data['first_pass']
        second_pass_boxes = data['second_pass']
    
    final_boxes = []
    
    # 处理第一次检测的框（绿色）
    first_pass_filtered = []
    for i, box1 in enumerate(first_pass_boxes):
        x1, y1, x2, y2 = map(int, box1)
        keep_box = True
        
        # 检查与其他第一次检测的框是否重叠
        for j, box2 in enumerate(first_pass_boxes):
            if i != j and boxes_intersect(box1, box2):
                # 比较框的大小，保留大的框
                area1 = (x2 - x1) * (y2 - y1)
                x1_2, y1_2, x2_2, y2_2 = map(int, box2)
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                if area1 < area2:  # 如果当前框更小，就不保留
                    keep_box = False
                    break
        
        # 检查是否与第二次检测的框重叠（第二次检测优先）
        for second_box in second_pass_boxes:
            if boxes_intersect(box1, second_box):
                keep_box = False
                break
        
        if keep_box:
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 225), 2)  # 绿色
            first_pass_filtered.append(box1)
            final_boxes.append(box1)
    
    # 处理第二次检测的框（红色）- 也处理重复框
    second_pass_filtered = []
    for i, box1 in enumerate(second_pass_boxes):
        x1, y1, x2, y2 = map(int, box1)
        keep_box = True
        
        # 检查与其他第二次检测的框是否重叠
        for j, box2 in enumerate(second_pass_boxes):
            if i != j and boxes_intersect(box1, box2):
                # 比较框的大小，保留大的框
                area1 = (x2 - x1) * (y2 - y1)
                x1_2, y1_2, x2_2, y2_2 = map(int, box2)
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                if area1 < area2:  # 如果当前框更小，就不保留
                    keep_box = False
                    break
        
        if keep_box:
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色
            second_pass_filtered.append(box1)
            final_boxes.append(box1)
    
    # 保存带框的图片
    output_dir = os.path.dirname(boxes_json_path)
    output_path = os.path.join(output_dir, 'result_with_boxes.tif')
    cv2.imwrite(output_path, output_img)
    print(f"Saved result to: {output_path}")
    
    # 保存最终的框信息
    with open(final_boxes_path, 'w') as f:
        json.dump({'boxes': final_boxes}, f)