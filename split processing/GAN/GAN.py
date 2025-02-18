import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
import rasterio
from gfpgan import GFPGANer

def read_tif(tif_path):
    """读取tif文件并转换为适合GFPGAN处理的格式"""
    try:
        with rasterio.open(tif_path) as src:
            # 读取数据
            image = src.read(1)  # 读取第一个波段
            
            # 确保数据类型正确
            image = image.astype(np.float32)
            
            # 数据标准化到0-255
            percentile_min = np.percentile(image, 1)
            percentile_max = np.percentile(image, 99)
            image_normalized = np.clip(
                ((image - percentile_min) * 255 / (percentile_max - percentile_min)),
                0, 255
            ).astype(np.uint8)
            
            # 转换为3通道
            image_rgb = cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2BGR)
            
            print(f"Successfully read TIF file: {tif_path}")
            print(f"Image shape: {image_rgb.shape}")
            print(f"Value range: {image_rgb.min()} - {image_rgb.max()}")
            
            return image_rgb, src.profile
    except Exception as e:
        print(f"Error reading TIF file: {e}")
        return None, None

def save_tif(img, profile, save_path):
    """保存处理后的图像为tif格式"""
    try:
        # 转换回单通道
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 更新profile
        profile.update({
            'count': 1,
            'height': img_gray.shape[0],
            'width': img_gray.shape[1],
            'dtype': 'uint8'
        })
        
        # 保存为tif
        with rasterio.open(save_path, 'w', **profile) as dst:
            dst.write(img_gray, 1)
            
        print(f"Successfully saved TIF file: {save_path}")
        return True
    except Exception as e:
        print(f"Error saving TIF file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='Fig\\input\\split_r1_c2_x13386_y6913_s2000.tif',
        help='Input image or folder')
    parser.add_argument(
        '-o', 
        '--output', 
        type=str, 
        default='REF_pic\\6', 
        help='Output folder')
    parser.add_argument(
        '-v', 
        '--version', 
        type=str, 
        default='1.3', 
        help='GFPGAN model version')
    parser.add_argument(
        '-s', 
        '--upscale', 
        type=int, 
        default=2, 
        help='The final upsampling scale of the image')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, ''), exist_ok=True)

    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    # 使用正确的权重路径
    model_path = 'sax/GFPGAN-master/experiments/pretrained_models/GFPGANv1.3.pth'
    
    # 使用 clean 架构，但只用它的超分辨率部分
    restorer = GFPGANer(
        model_path=model_path,
        upscale=args.upscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None)

    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*.tif')))

    for img_path in img_list:
        print(f'\nProcessing: {img_path}')
        
        input_img, profile = read_tif(img_path)
        if input_img is None:
            continue
            
        # 如果是单通道图像，转换为三通道
        if len(input_img.shape) == 2:
            input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        
        # 保存原始图像的统计信息
        mean_before = np.mean(input_img, axis=(0, 1))
        std_before = np.std(input_img, axis=(0, 1))
        
        # 处理图像，设置 weight=0 只做超分辨率
        _, _, restored_img = restorer.enhance(
            input_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.0)  # 设置为 0 表示不做增强，只做超分辨率

        if restored_img is not None:
            # 调整回原始的统计特征
            mean_after = np.mean(restored_img, axis=(0, 1))
            std_after = np.std(restored_img, axis=(0, 1))
            
            # 恢复原始亮度和对比度
            restored_img = ((restored_img - mean_after) * (std_before / std_after) + mean_before).clip(0, 255).astype(np.uint8)
            
            basename = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(args.output, f'{basename}_restored.tif')
            save_tif(restored_img, profile, save_path)

    print(f'Results are in the [{args.output}] folder.')

if __name__ == '__main__':
    main()
