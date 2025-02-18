import rasterio
import numpy as np
from pathlib import Path

def split_tif_3x6(tif_path, output_dir="MoonCode\\material\\Study region"):
    """
    将tif图像精确分割成3行6列
    
    Args:
        tif_path: tif文件路径
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
        block_h = height // 3  # 3行
        block_w = width // 6   # 6列
        
        print(f"每个分块大小: {block_w}x{block_h}")
        
        # 分割并保存
        for i in range(3):  # 3行
            for j in range(6):  # 6列
                # 计算当前块的坐标
                y_start = i * block_h
                x_start = j * block_w
                
                # 读取数据块
                window = rasterio.windows.Window(x_start, y_start, block_w, block_h)
                block = src.read(window=window)
                
                # 生成输出文件名
                output_name = f"split_r{i}_c{j}.tif"
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
                
                print(f"保存分块 {i}_{j} 到 {output_path}")
                print(f"起始坐标: ({x_start}, {y_start})")
        
        print(f"\n分割完成!")
        print(f"总共生成 18 个分块")

if __name__ == "__main__":
    # tif文件路径
    tif_path = r"MoonCode\\material\\Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
    
    # 执行分割
    split_tif_3x6(tif_path)
