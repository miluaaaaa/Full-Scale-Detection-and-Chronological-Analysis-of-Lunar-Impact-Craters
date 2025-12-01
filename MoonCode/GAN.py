import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
import rasterio
import torch.nn as nn



class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23):
        super(RRDBNet, self).__init__()
        from basicsr.archs.rrdbnet_arch import RRDBNet as RRDB  # 复用 basicsr
        
        self.model = RRDB(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb)

    def forward(self, x):
        return self.model(x)



def read_tif(tif_path):
    try:
        with rasterio.open(tif_path) as src:
            image = src.read(1).astype(np.float32)

            p_min = np.percentile(image, 1)
            p_max = np.percentile(image, 99)
            image_norm = np.clip((image - p_min) * 255 / (p_max - p_min), 0, 255).astype(np.uint8)

            image_rgb = cv2.cvtColor(image_norm, cv2.COLOR_GRAY2BGR)

            print(f"Read TIF: {tif_path}")
            return image_rgb, src.profile
    except Exception as e:
        print(f"Error reading: {e}")
        return None, None



def save_tif(img, profile, save_path):
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        profile.update({
            'count': 1,
            'height': img_gray.shape[0],
            'width': img_gray.shape[1],
            'dtype': 'uint8'
        })

        with rasterio.open(save_path, 'w', **profile) as dst:
            dst.write(img_gray, 1)

        print(f"Saved TIF: {save_path}")
        return True
    except Exception as e:
        print(f"Error saving: {e}")
        return False



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='GAN_Supplementary Experiment/Before_GAN/4')
    parser.add_argument('-o', '--output', type=str,
                        default='GAN_Supplementary Experiment/After_GAN/4')
    parser.add_argument('-w', '--weights', type=str,
                        default='GAN_Supplementary Experiment/weights/ESRGAN_G.pth')
    parser.add_argument('-s', '--upscale', type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    generator = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23).to(device)


    print(f"Loading pretrained GAN weights: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    generator.load_state_dict(checkpoint, strict=True)
    generator.eval()


    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*.tif')))

    for img_path in img_list:
        print(f"\nProcessing: {img_path}")

        input_img, profile = read_tif(img_path)
        if input_img is None:
            continue


        mean_before = np.mean(input_img)
        std_before = np.std(input_img)


        img_tensor = torch.from_numpy(input_img / 255.).permute(2, 0, 1).float().unsqueeze(0).to(device)


        with torch.no_grad():
            output = generator(img_tensor)

        restored_img = (output.squeeze().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        mean_after = restored_img.mean()
        std_after = restored_img.std()
        restored_img = ((restored_img - mean_after) * (std_before / std_after) + mean_before).clip(0, 255).astype(np.uint8)

        basename = os.path.splitext(os.path.basename(img_path))[0]
        save_tif(restored_img, profile, os.path.join(args.output, f"{basename}_GAN.tif"))

    print(f"\nDone! Results saved in: {args.output}")


if __name__ == '__main__':
    main()
