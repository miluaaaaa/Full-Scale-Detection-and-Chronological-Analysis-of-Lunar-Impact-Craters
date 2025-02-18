import os
from plotting import process_small_images, process_large_image, process_and_draw_boxes, generate_csv_from_json
from pickup import merge_images_with_boxes

def main(test_num, first_conf=0.3, second_conf=0.15, scale_factor=2.0):
    """主函数"""
    try:
        # 配置参数
        params = {
            'first_conf': first_conf,    # 第一次检测的置信度阈值
            'second_conf': second_conf,  # 第二次检测的置信度阈值
            'max_det': 1000,
            'line_thickness': 2,
            'scale_factor': 50.0000209  # m/pixel，GAN处理后的新比例
        }
        
        print(f"\nProcessing test {test_num}...")
        print(f"First pass confidence threshold: {first_conf}")
        print(f"Second pass confidence threshold: {second_conf}")
        
        # 设置路径
        model_path = 'yolov8_ADown.pt'
        small_images_folder = 'Fig\\input\\64sg'  # 修正小图片路径
        ref_image_path = 'REF_pic\\6\\split_r1_c2_x13386_y6913_s2000_restored.tif'
        
        # 确保输出文件夹存在
        os.makedirs(f'{test_num}t', exist_ok=True)
        
        boxes_json_path = f'{test_num}t/boxes.json'
        final_boxes_path = f'{test_num}t/final_boxes.json'
        csv_path = f'{test_num}t/craters_report.csv'
        
        # 第一步：处理小图和参考图，生成boxes.json
        print(f"\nProcessing test {test_num}...")
        print("\nStep 1: Processing images...")
        print(f"Model path: {model_path}")
        print(f"Small images folder: {small_images_folder}")
        print(f"Reference image: {ref_image_path}")
        print(f"First pass confidence threshold: {first_conf}")
        print(f"Second pass confidence threshold: {second_conf}")
        print(f"Maximum detections: {params['max_det']}")
        
        process_small_images(model_path, small_images_folder, ref_image_path, boxes_json_path, params)
        
        # 第二步：画框并保存结果
        print("\nStep 2: Drawing boxes...")
        print(f"Drawing results to: {ref_image_path}")
        print(f"Using boxes from: {boxes_json_path}")
        print(f"Line thickness: {params['line_thickness']}")
        
        process_and_draw_boxes(ref_image_path, boxes_json_path, final_boxes_path)
        
        # 第三步：生成CSV报告（包含直径信息）
        print("\nStep 3: Generating CSV report...")
        print(f"Saving report to: {csv_path}")
        
        generate_csv_from_json(final_boxes_path, csv_path, params)
        
        print("\nAll steps completed successfully!")
        print(f"Results saved in folder: {test_num}t/")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_num = 64
    # 修改调用方式，使用两个明确的置信度参数
    main(test_num, first_conf=0.1, second_conf=0.3)