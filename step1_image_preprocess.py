import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from  multiprocessing import Process,Pool
# 1. 调整窗宽窗位函数
def apply_windowing(image_data, window_width, window_level):
    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)
    
    # 限制灰度级别到窗宽窗位的范围
    windowed_image = np.clip(image_data, lower_bound, upper_bound)
    
    # 归一化到 [0, 255]
    windowed_image = ((windowed_image - lower_bound) / window_width) * 255.0
    windowed_image = windowed_image.astype(np.uint8)
    
    return windowed_image

# 2. 归一化函数（Z-score标准化）
def normalize_image(image_data):
    mean = np.mean(image_data)
    std = np.std(image_data)
    return (image_data - mean) / std

# 3. 重采样函数
def resample_image(image, target_spacing=(1.0, 1.0, 1.0)):
    sitk_image = sitk.GetImageFromArray(image)
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing(target_spacing)

    new_size = [
        int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]

    resample.SetSize(new_size)
    resampled_image = resample.Execute(sitk_image)

    return sitk.GetArrayFromImage(resampled_image)

# 4. 处理单个图像文件
def process_image(file_path, output_dir, window_width, window_level, target_spacing):
    # 加载NIfTI图像
    img = nib.load(file_path)
    image_data = img.get_fdata()

    # 调整窗宽窗位
    windowed_image = apply_windowing(image_data, window_width, window_level)

    # 归一化
    normalized_image = normalize_image(windowed_image)

    # 重采样
    resampled_image = resample_image(normalized_image, target_spacing)

    # 保存预处理后的图像
    output_filename = os.path.join(output_dir, os.path.basename(file_path))
    processed_img = nib.Nifti1Image(resampled_image, img.affine)
    nib.save(processed_img, output_filename)
    print(f"Processed and saved: {output_filename}")

# 5. 批量处理文件夹中的所有文件
def process_folder(input_dir, output_dir, window_width=400, window_level=50, target_spacing=(1.0, 1.0, 1.0)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pool2=Pool(15)
    for filename in os.listdir(input_dir):
        if filename.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, filename)
            pool2.apply_async(func=process_image,args=(file_path, output_dir, window_width, window_level, target_spacing))
    pool2.close()
    pool2.join()

if __name__ == '__main__':

    import sys
    # 示例执行代码
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    window_width = 350  # 调整腹部软组织的窗宽
    window_level = 50   # 调整腹部软组织的窗位
    target_spacing = (1.0, 1.0, 1.0)  # 目标分辨率
    process_folder(input_dir,output_dir,window_width,window_level,target_spacing)
