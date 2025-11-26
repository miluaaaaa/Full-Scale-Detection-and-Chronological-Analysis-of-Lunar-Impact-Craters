import rasterio
import numpy as np
from pathlib import Path
import json

def split_tif(tif_path, rows, cols, output_dir="split_output"):
    """
    将tif图像分割成指定的行数和列数
    
    Args:
        tif_path: tif文件路径
        rows: 需要分割的行数
        cols: 需要分割的列数
        output_dir: 输出目录
    """
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    with rasterio.open(tif_path) as src:
        # 获取图像信息
        height = src.height
        width = src.width
        
        print(f"原始图像大小: {width}x{height}")
        
        # 计算每个块的大小
        block_h = height // rows
        block_w = width // cols
        
        print(f"每个分块大小: {block_w}x{block_h}")
        
        # 生成文件名基础
        base_name = Path(tif_path).stem
        
        # 分割并保存
        splits_info = []
        for i in range(rows):
            for j in range(cols):
                # 计算当前块的坐标
                y_start = i * block_h
                x_start = j * block_w
                
                # 定义窗口
                window = rasterio.windows.Window(x_start, y_start, block_w, block_h)
                
                # 读取数据块
                block = src.read(window=window)
                
                # 生成输出文件名
                output_name = f"{base_name}_r{i}_c{j}.tif"
                output_path = Path(output_dir) / output_name
                
                # 更新变换矩阵
                transform = rasterio.windows.transform(window, src.transform)
                
                # 创建新的tif文件
                profile = src.profile.copy()
                profile.update({
                    'height': block_h,
                    'width': block_w,
                    'transform': transform
                })
                
                # 保存分块
                with rasterio.open(str(output_path), 'w', **profile) as dst:
                    dst.write(block)
                
                # 记录信息
                block_info = {
                    'row': i,
                    'col': j,
                    'x_start': x_start,
                    'y_start': y_start,
                    'width': block_w,
                    'height': block_h,
                    'output_path': str(output_path)
                }
                splits_info.append(block_info)
                
                print(f"保存分块 {i}_{j} 到 {output_path}")
                print(f"起始坐标: ({x_start}, {y_start})")
        
        # 保存分割信息
        info_path = Path(output_dir) / f"{base_name}_splits_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(splits_info, f, indent=4, ensure_ascii=False)
        
        print(f"\n分割完成!")
        print(f"总共生成 {rows*cols} 个分块")
        print(f"分块信息已保存到: {info_path}")
        
        return splits_info

if __name__ == "__main__":
    # 用户输入
    tif_path = input("请输入tif文件路径: ")
    rows = int(input("请输入要分割的行数: "))
    cols = int(input("请输入要分割的列数: "))
    output_dir = input("请输入输出目录名称 (直接回车使用默认'split_output'): ") or "split_output"
    
    # 执行分割
    split_tif(tif_path, rows, cols, output_dir)
