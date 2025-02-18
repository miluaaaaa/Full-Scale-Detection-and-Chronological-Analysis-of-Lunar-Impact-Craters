import os
import cv2
import numpy as np
import json
import re

def extract_coordinates(filename):
    """从文件名中提取坐标信息"""
    pattern = r'x(\d+)_y(\d+)'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def check_pixel_overlap(img1, box1, img2, box2):
    """检查两个框的红色像素是否有重叠"""
    x1_1, y1_1, x2_1, y2_1 = map(int, box1)
    x1_2, y1_2, x2_2, y2_2 = map(int, box2)
    
    # 获取两个框的区域
    roi1 = img1[y1_1:y2_1, x1_1:x2_1]
    roi2 = img2[y1_2:y2_2, x1_2:x2_2]
    
    # 检查是否有红色像素 (BGR格式中红色是[0,0,255])
    red_pixels1 = np.any((roi1 == [0,0,255]).all(axis=2))
    red_pixels2 = np.any((roi2 == [0,0,255]).all(axis=2))
    
    return red_pixels1 and red_pixels2

def merge_images_with_boxes(input_folder, output_path, boxes_json_path, params):
    """拼接图片，并添加边界融合"""
    # Add debug prints for box loading
    print("\nDEBUG: Loading boxes from JSON")
    with open(boxes_json_path, 'r') as f:
        boxes_data = json.load(f)
        print(f"Loaded first pass boxes: {len(boxes_data['first_pass'])}")
        print(f"Loaded second pass boxes: {len(boxes_data['second_pass'])}")

    image_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    
    try:
        images = []
        for filename in image_files:
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img))
        
        if not images:
            return None
            
        tile_height, tile_width = images[0][1].shape[:2]
        rows = int(np.ceil(np.sqrt(len(images))))
        cols = rows
        canvas = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=images[0][1].dtype)
        
        # 添加边界融合
        overlap = 10  # 重叠像素数
        alpha = np.linspace(0, 1, overlap)
        
        for idx, (filename, img) in enumerate(images):
            if idx >= rows * cols:
                break
                
            row = idx // cols
            col = idx % cols
            y_start = row * tile_height
            x_start = col * tile_width
            
            # 处理重叠区域
            if row > 0:  # 上边界融合
                for i in range(overlap):
                    weight = alpha[i]
                    canvas[y_start+i, x_start:x_start+tile_width] = \
                        (1-weight) * canvas[y_start+i, x_start:x_start+tile_width] + \
                        weight * img[i, :]
                    
            if col > 0:  # 左边界融合
                for i in range(overlap):
                    weight = alpha[i]
                    canvas[y_start:y_start+tile_height, x_start+i] = \
                        (1-weight) * canvas[y_start:y_start+tile_height, x_start+i] + \
                        weight * img[:, i]
            
            # 放置主要图像区域
            y_end = min(y_start + tile_height, canvas.shape[0])
            x_end = min(x_start + tile_width, canvas.shape[1])
            canvas[y_start:y_end, x_start:x_end] = img[:y_end-y_start, :x_end-x_start]
        
        # 保存结果
        cv2.imwrite(output_path, canvas)
        
        return canvas
        
    except Exception as e:
        print(f"Error in merge_images_with_boxes: {str(e)}")
        raise

def filter_boxes(boxes, scores):
    """按照检测顺序过滤框，如果新框与旧框相交，保留新框"""
    if len(boxes) == 0:
        return [], []
    
    keep = []
    keep_scores = []
    
    # 按照检测顺序遍历（保持原始顺序）
    for i in range(len(boxes)):
        current_box = boxes[i]
        current_score = scores[i]
        
        # 找出与当前框相交的旧框的索引
        intersect_indices = []
        for j, idx in enumerate(keep):
            if boxes_intersect(current_box, boxes[idx]):
                intersect_indices.append(j)
        
        # 删除所有相交的旧框
        for j in reversed(intersect_indices):
            keep.pop(j)
            keep_scores.pop(j)
        
        # 添加新框
        keep.append(i)
        keep_scores.append(current_score)
    
    return boxes[keep], np.array(keep_scores)

def boxes_intersect(box1, box2):
    """检查两个框是否相交"""
    x1_1, y1_1, x2_1, y2_1 = map(int, box1)
    x1_2, y1_2, x2_2, y2_2 = map(int, box2)
    
    # 如果一个框在另一个框的完全左边或右边，则不相交
    if x2_1 < x1_2 or x1_1 > x2_2:
        return False
    # 如果一个框在另一个框的完全上边或下边，则不相交
    if y2_1 < y1_2 or y1_1 > y2_2:
        return False
    return True

def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = map(int, box1)
    x1_2, y1_2, x2_2, y2_2 = map(int, box2)
    
    # 计算交集区域
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算两个框的面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 计算IoU
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def nms_filter(boxes, scores, iou_threshold=0.3):
    """使用NMS算法过滤重复框"""
    if len(boxes) == 0:
        return [], []
        
    # 按置信度排序
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    scores = scores[indices]
    
    keep = []
    
    while len(indices) > 0:
        keep.append(indices[0])
        
        ious = np.array([calculate_iou(boxes[indices[0]], box) for box in boxes[1:]])
        indices = indices[1:][ious < iou_threshold]
    
    return boxes[keep], scores[keep]

if __name__ == "__main__":
    # 这部分代码将不会被直接使用，因为我们会通过main.py调用函数
    pass
