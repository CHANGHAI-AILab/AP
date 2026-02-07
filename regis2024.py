import os
import sys
import glob
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from multiprocessing import Pool
def resize_image_itk_mapping(input_file,target_img):
    # 读取NIfTI文件
    img = nib.load(input_file)
    data = img.get_fdata()

    data_target = sitk.ReadImage(target_img)

    # 定义目标维度
    target_shape = (data_target.GetSize()[0], data_target.GetSize()[1], data_target.GetSize()[2])

    # 计算缩放因子
    zoom_factors = [
        target_shape[0] / data.shape[0],
        target_shape[1] / data.shape[1],
        target_shape[2] / data.shape[2]
    ]
    # 进行缩放
    resampled_data = zoom(data, zoom_factors, order=1)  # order=1 代表线性插值
    # 创建新的NIfTI图像
    new_img = nib.Nifti1Image(resampled_data, img.affine, img.header)
    # 保存新文件
    nib.save(new_img, input_file)


def task(i,all_img,all_mask,all_fixed,fixed_masks):
    if i in all_mask and i in all_fixed and i :
        if i+'_deformed_seg.nii.gz' not in os.listdir(fixed_masks):
            move_img = os.path.join(move_imgs, i)
            move_mask = os.path.join(move_masks, i)
            fixed_img = os.path.join(fixed_imgs, i)
            fixed_mask = os.path.join(fixed_masks, i)
            fixed = sitk.ReadImage(fixed_img)
            moving = sitk.ReadImage(move_img)

            #print(moving)


            if fixed.GetSize()[2]==moving.GetSize()[2]:
                pass
                #os.system(f'/data/deedsBCV/deedsBCV -F {fixed_img} -M {move_img}  -O {fixed_mask} -S {move_mask}')
            else:
                img_r = resize_image_itk_mapping(fixed_img,move_img)
                #sitk.WriteImage(img_r, fixed_img)
                os.system(f'/data2/XB/deedsBCV-master/deedsBCV -F {fixed_img} -M {move_img}  -O {fixed_mask} -S {move_mask}')
        else:
            print(f'not found {i}')


def run(move_imgs,move_masks,fixed_imgs,fixed_masks):
    all_img = os.listdir(move_imgs)
    all_mask = os.listdir(move_masks)
    all_fixed = os.listdir(fixed_imgs)

    pool2=Pool(7)
    for i in all_img:
        task(i,all_img,all_mask,all_fixed,fixed_masks)
        #pool2.apply_async(func=task,args=(i,all_img,all_mask,all_fixed,fixed_masks))
    pool2.close()
    pool2.join()

            
move_imgs = sys.argv[1]
move_masks = sys.argv[2]

fixed_imgs = sys.argv[3]

fixed_masks = fixed_imgs + '_mask'
os.makedirs(fixed_masks, exist_ok=True)

run(move_imgs, move_masks, fixed_imgs, fixed_masks)
