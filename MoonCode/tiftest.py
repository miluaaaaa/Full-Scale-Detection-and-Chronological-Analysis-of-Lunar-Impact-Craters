import rasterio
import numpy as np

def read_moon_tif():
    # 读取tif文件
    tif_path = r"MoonCode\material\Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
    
    with rasterio.open(tif_path) as src:
        # 读取基本信息
        print(f"图像大小: {src.width} x {src.height}")  # 109164 x 54582
        print(f"像素分辨率: {src.res[0]} 米")  # 约100米
        print(f"坐标系统: {src.crs}")
        print(f"波段数量: {src.count}")
        
        # 读取图像数据
        image = src.read(1)  # 读取第一个波段
        
        # 输出一些基本统计信息
        print(f"\n图像统计信息:")
        print(f"最小值: {image.min()}")
        print(f"最大值: {image.max()}")
        print(f"平均值: {image.mean():.2f}")
        
        return image

if __name__ == "__main__":
    moon_image = read_moon_tif()
    
    # 如果需要保存一个小区域进行测试
    # test_region = moon_image[1000:2000, 1000:2000]
    # np.save("test_region.npy", test_region)
