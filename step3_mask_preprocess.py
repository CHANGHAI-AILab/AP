import nibabel as nib
import numpy as np
import os
from multiprocessing import Pool
import concurrent.futures
import nibabel as nib
import SimpleITK as sitk
import shutil

def process_mask_file(mask_path, output_path):
    # Load the mask file
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    # Print details before processing
    unique_labels_before = np.unique(mask_data)

    # Convert labels to integer
    mask_data_int = mask_data.astype(np.int32)

    #mask_data_int[mask_data_int>0]=1
    mask_data_int[mask_data_int!=2]=0
    mask_data_int[mask_data_int==2]=1

    # Print details after processing
    unique_labels_after = np.unique(mask_data_int)

    # Save the new mask file
    new_mask_img = nib.Nifti1Image(mask_data_int, mask_img.affine, mask_img.header)
    nib.save(new_mask_img, output_path)


def convert_labels_to_int(ct_mask_dir, output_dir, num_workers=20):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = []
    pool2=Pool(num_workers)

    for mask_filename in os.listdir(ct_mask_dir):
        

        if mask_filename.endswith('.nii.gz'):
            mask_path = os.path.join(ct_mask_dir, mask_filename)
            output_path = os.path.join(output_dir, mask_filename)
            tasks.append((mask_path, output_path))
            pool2.apply_async(func=process_mask_file,args=(mask_path, output_path))
    pool2.close()
    pool2.join()



def check_mask_file(args):
    filepath, valid_labels = args
    mask_data = nib.load(filepath).get_fdata()
    file_labels = np.unique(mask_data)

    invalid_labels = set(file_labels) - valid_labels
    if  len(file_labels)==1:
        invalid_labels=set([0])
    if invalid_labels:
        return filepath, invalid_labels
    return None

def check_mask_labels(mask_folder, num_workers=20):
    valid_labels = {0, 1}
    invalid_masks = []

    # 准备要处理的文件列表
    tasks = []
    for filename in os.listdir(mask_folder):


        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            filepath = os.path.join(mask_folder, filename)
            tasks.append((filepath, valid_labels))

    # 使用Pool并行处理任务
    with Pool(num_workers) as pool:
        results = pool.map(check_mask_file, tasks)

    # 收集有问题的文件
    invalid_masks = [result for result in results if result]
    return invalid_masks


def find_empty_masks(ct_dir, mask_dir):
    # 获取所有 CT 和 mask 文件
    ct_files = [f for f in os.listdir(ct_dir) if f.endswith('.nii.gz')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')]
    
    empty_masks = []

    # 遍历所有 CT 文件
    for ct_file in ct_files:
        # 对应的 mask 文件
        mask_file = ct_file  # 假设 mask 文件与 CT 文件同名
        
        if mask_file in mask_files:
            mask_path = os.path.join(mask_dir, mask_file)
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata()


            # 检查 mask 是否全为零
            if np.all(mask_data == 0):
                empty_masks.append(ct_file)
    
    return empty_masks



def process_mask_file2(ct_path, mask_path, output_path, label_to_convert, pool_size=4):
    """
    Process a mask file to change a specified label to 1, while keeping label 0 and changing other labels to 0.
    """
    ct_img = nib.load(ct_path)
    mask_img = nib.load(mask_path)
    
    mask_data = mask_img.get_fdata()
    
    new_mask_data = np.where(mask_data == label_to_convert, 1, np.where(mask_data == 0, 0, 0))
    
    new_mask_img = nib.Nifti1Image(new_mask_data, mask_img.affine, mask_img.header)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    output_file = os.path.join(output_path, os.path.basename(mask_path))
    nib.save(new_mask_img, output_file)


def main(ct_dir, mask_dir, output_dir, label_to_convert, pool_size=4):
    """
    Main function to process all mask files in the specified directory with multithreading.
    """
    ct_files = sorted([os.path.join(ct_dir, f) for f in os.listdir(ct_dir) if f.endswith('.nii.gz')])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
    
    assert len(ct_files) == len(mask_files), "Mismatch between number of CT and mask files"
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = [executor.submit(process_mask_file2, ct_files[i], mask_files[i], output_dir, label_to_convert) for i in range(len(ct_files))]
        concurrent.futures.wait(futures)


def run(mask_path):
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    unique_labels_before = np.unique(mask_data)
    if 1 in unique_labels_before and 2 in unique_labels_before:
        pass
    else:
        print('no',mask_path)


def match(tumor_mask,vessel_mask,output):
    tumor_img = nib.load(tumor_mask)
    vessel_img = nib.load(vessel_mask)

    # 获取数据和仿射矩阵
    tumor_data = tumor_img.get_fdata()
    vessel_data = vessel_img.get_fdata()
    tumor_affine = tumor_img.affine
    vessel_affine = vessel_img.affine

    # 检查仿射矩阵是否一致
    if not np.allclose(tumor_affine, vessel_affine):
        print('not match affine_transform',tumor_mask)
        # 计算仿射变换矩阵，将血管 mask 对齐到肿瘤 mask 空间
        affine_transform_matrix = np.linalg.inv(vessel_affine) @ tumor_affine
        
        # 使用仿射矩阵对齐血管数据
        aligned_vessel_data = affine_transform(vessel_data, affine_transform_matrix[:3, :3], offset=affine_transform_matrix[:3, 3], output_shape=tumor_data.shape)
    else:
        # 如果已经对齐，则直接使用原始血管数据
        aligned_vessel_data = vessel_data
    aligned_vessel_img = nib.Nifti1Image(aligned_vessel_data.astype(np.float32), tumor_affine)
    os.makedirs(output,exist_ok=True)

    nib.save(aligned_vessel_img, output+'\\'+vessel_mask.split('\\')[-1])
    print(output+'\\'+vessel_mask.split('\\')[-1])
    return output+'\\'+vessel_mask.split('\\')[-1]


import nibabel as nib
from nilearn.image import resample_img
from radiomics import featureextractor
import numpy as np

def load_nifti(file_path):
    """加载 NIfTI 文件"""
    return nib.load(file_path)




    # 打印所有特征
    for feature_name, value in features.items():
        print(f"{feature_name}: {value}")

def check_and_fix_geometry(image_path, mask_path, output_mask_path):
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
        
    # 重采样掩膜以匹配图像
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(image.GetSize())
    resampler.SetOutputSpacing(image.GetSpacing())
    resampler.SetDefaultPixelValue(0)
    resampled_mask = resampler.Execute(mask)

    # 保存重采样后的掩膜
    sitk.WriteImage(resampled_mask, output_mask_path)
    
        

if __name__ == '__main__':

    import sys
    import glob

    ct_directory=sys.argv[1]
    ct_mask_dir=sys.argv[2]
    output_dir=sys.argv[3]
    p = Pool(40)

    # 1111111Example usage:
    #ct_mask_dir = r'G:\sjjr_20240823_feature\zhongshan\\ZSCT_nii_tumor'
    #output_dir = r'G:\sjjr_20240823_feature\zhongshan\\ZSCT_nii_tumor_int'
    """
    os.makedirs(output_dir,exist_ok=True)
    for each_t in glob.glob(ct_directory+'\\*.nii.gz'):
        each_mask=os.path.join(ct_mask_dir,each_t.split('\\')[-1])
        p.apply_async(match, args=(each_t,each_mask,output_dir,))
    p.close()
    p.join()


    """
    
    convert_labels_to_int(ct_mask_dir, output_dir, num_workers=20)

    # 2222222指定mask文件夹路径
    mask_folder_path = output_dir
    invalid_masks = check_mask_labels(mask_folder_path, num_workers=20)
    if invalid_masks:
        print("The following masks have invalid labels:")
        for filepath, labels in invalid_masks:
            print(f'{os.path.basename(filepath)}: {labels}')
    else:
        print("All masks have valid labels (0 and 1only).")

    """
    # 33333333设置 CT 和 mask 文件所在的目录
    #ct_directory = r'G:\sjjr_20240823_feature\zhongshan\ZSCT_nii'
    mask_directory = ct_mask_dir
    # 找出所有没有 label 的 mask
    empty_mask_files = find_empty_masks(ct_directory, mask_directory)
    # 打印结果
    print("没有 label 的样本名:")
    for mask in empty_mask_files:
        print(mask)
    """
    """
    # 44444444CT 和 mask 文件所在的目录
    # Directories containing CT and mask files, and output directory
    ct_directory = "F:\\syx_lwb_project\\c_demo\\C_niigz_114_machine_doing"
    mask_directory = "F:\\syx_lwb_project\\c_demo\\c_mask"
    output_directory = "F:\\syx_lwb_project\\c_demo\\c_label_int"
    # Label to convert
    label_to_change = 4
    # Run the main function
    main(ct_directory, mask_directory, output_directory, label_to_change, pool_size=4)

    

    mask_folder=r"F:\syx_lwb_project\p_demo\p_fix_mask_p5"
    p = Pool(20)
    for mask_file in os.listdir(mask_folder):
        mask_path = os.path.join(mask_folder, mask_file)
        #run(mask_path)
        p.apply_async(run, args=(mask_path,))
    p.close()
    p.join()
    """





