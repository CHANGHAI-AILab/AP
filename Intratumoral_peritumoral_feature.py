import os
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool
import os, time, random


def load_image(image_path):
    """读取CT或mask图像"""
    return sitk.ReadImage(image_path)

def save_image(image, output_path):
    """保存处理后的mask图像"""
    sitk.WriteImage(image, output_path)

def expand_roi(mask_image, expansion_mm):
    """对mask的ROI区域进行外扩并保留原label1区域"""
    # 获取体素大小（spacing）
    spacing = mask_image.GetSpacing()
    
    # 计算2mm对应的体素数量
    expansion_voxels = [int(expansion_mm / sp) for sp in spacing]
    
    # 转换为NumPy数组
    mask_array = sitk.GetArrayFromImage(mask_image)
    
    # 获取label1区域
    label1_region = (mask_array == 1)
    
    # 创建SimpleITK的label1图像
    label1_image = sitk.GetImageFromArray(label1_region.astype(np.uint8))
    label1_image.CopyInformation(mask_image)
    
    # 使用BinaryDilate扩展label1区域，但保留原始的label1区域
    dilated_label1 = sitk.BinaryDilate(label1_image, expansion_voxels, sitk.sitkBall)
    
    # 转换回NumPy数组
    dilated_array = sitk.GetArrayFromImage(dilated_label1)
    
    # 创建新mask，其中扩展的区域为label2，原有的label1保持不变
    # 扩展区域是扩展后的区域减去原有label1区域，确保label2不会覆盖label1
    label2_region = np.logical_and(dilated_array == 1, mask_array != 1)
    
    # 将原label1区域保留，并将扩展部分标记为label2
    new_mask_array = np.where(label2_region, 2, mask_array)
    
    # 将数组转换回SimpleITK图像
    new_mask_image = sitk.GetImageFromArray(new_mask_array)
    new_mask_image.CopyInformation(mask_image)
    
    return new_mask_image


def run(mask_path,expansion_mm,output_folder,mask_file):
    # 读取mask
    mask_image = load_image(mask_path)
    # 扩展mask的ROI区域
    expanded_mask = expand_roi(mask_image, expansion_mm)
    # 保存处理后的mask
    output_path = os.path.join(output_folder, mask_file)
    save_image(expanded_mask, output_path)
    print(f"Processed {mask_file}, saved to {output_path}")




def process_masks(mask_folder, output_folder, expansion_mm=5):
    """遍历mask文件夹，处理每个mask"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    mask_files = os.listdir(mask_folder)
    p = Pool(20)
    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        #run(mask_path,expansion_mm,output_folder)
        if os.path.exists(os.path.join(output_folder, mask_file)):
            continue
        p.apply_async(run, args=(mask_path,expansion_mm,output_folder,mask_file,))
    p.close()
    p.join()


if __name__ == "__main__":
    import sys
    mask_folder=sys.argv[1]
    output_folder=sys.argv[2]
    expansion_mm= int(sys.argv[3])

    #mask_folder = r"G:\sjjr_20240823_feature\one_total_xueguan_mask"  # 替换为mask图像的文件夹路径
    #output_folder = r"G:\sjjr_20240823_feature\one_total_xueguan_mask_p2"  # 替换为保存新mask的文件夹路径
    #expansion_mm = 2  # 扩展2mm
    process_masks(mask_folder, output_folder, expansion_mm)
