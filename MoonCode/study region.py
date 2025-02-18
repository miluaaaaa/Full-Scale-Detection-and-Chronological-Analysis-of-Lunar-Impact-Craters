import rasterio
import numpy as np
import random
from pathlib import Path

def crop_square_region_tif(tif_path, size_km=200, output_dir="Fig/input"):
    """
    从tif图像中随机截取200km×200km的正方形区域
    
    Args:
        tif_path: tif文件路径
        size_km: 需要截取的正方形边长(km)
        output_dir: 输出目录
    """
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    # 计算200km对应的像素数（100m/pixel）
    size_pixels = int(size_km * 1000 / 100)  # 2000 pixels
    
    with rasterio.open(tif_path) as src:
        # 获取图像信息
        height = src.height
        width = src.width
        
        print(f"原始图像大小: {width}x{height}")
        
        # 随机选择中心点（避开边缘）
        center_x = random.randint(size_pixels//2, width - size_pixels//2)
        center_y = random.randint(size_pixels//2, height - size_pixels//2)
        
        # 计算裁剪区域
        x_start = center_x - size_pixels//2
        y_start = center_y - size_pixels//2
        
        # 定义窗口
        window = rasterio.windows.Window(x_start, y_start, size_pixels, size_pixels)
        
        # 读取数据
        cropped = src.read(window=window)
        
        # 更新变换矩阵
        transform = rasterio.windows.transform(window, src.transform)
        
        # 生成输出文件名
        base_name = Path(tif_path).stem
        output_name = f"{base_name}_x{x_start}_y{y_start}_s{size_pixels}.tif"
        output_path = Path(output_dir) / output_name
        
        # 创建新的tif文件
        profile = src.profile.copy()
        profile.update({
            'height': size_pixels,
            'width': size_pixels,
            'transform': transform
        })
        
        # 保存裁剪后的图像
        with rasterio.open(str(output_path), 'w', **profile) as dst:
            dst.write(cropped)
        
        # 打印信息
        print(f"\n裁剪信息:")
        print(f"中心点坐标: ({center_x}, {center_y})")
        print(f"起始坐标: ({x_start}, {y_start})")
        print(f"尺寸: {size_pixels}x{size_pixels} pixels")
        print(f"保存到: {output_path}")
        
        return {
            'original_image': tif_path,
            'output_path': str(output_path),
            'center_x': center_x,
            'center_y': center_y,
            'x_start': x_start,
            'y_start': y_start,
            'size_pixels': size_pixels
        }

if __name__ == "__main__":
    # tif文件路径
    tif_path = r"MoonCode\\material\\chosen region\\split_r1_c2.tif"
    
    # 执行裁剪
    crop_info = crop_square_region_tif(tif_path)
