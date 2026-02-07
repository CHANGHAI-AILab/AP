import os
import sys
import glob
import SimpleITK as sitk

def resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkNearestNeighbor):
    """
    用itk方法将原始图像resample到与目标图像一致
    :param ori_img: 原始需要对齐的itk图像
    :param target_img: 要对齐的目标itk图像
    :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
    :return:img_res_itk: 重采样好的itk图像
    使用示范：
    import SimpleITK as sitk
    target_img = sitk.ReadImage(target_img_file)
    ori_img = sitk.ReadImage(ori_img_file)
    img_r = resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear)
    """
    target_Size = target_img.GetSize()      # 目标图像大小  [x,y,z]
    target_Spacing = target_img.GetSpacing()   # 目标的体素块尺寸    [x,y,z]
    target_origin = target_img.GetOrigin()      # 目标的起点 [x,y,z]
    target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_Size)      # 目标图像大小
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt8)   # 近邻插值用于mask的，保存uint8
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))    
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
    return itk_img_resampled

def run(move_imgs,move_masks,fixed_imgs,fixed_masks):
    all_img = os.listdir(move_imgs)
    all_mask = os.listdir(move_masks)
    all_fixed = os.listdir(fixed_imgs)

    for i in all_img:
        print(i)
        #try:
        if i in all_mask and i in all_fixed and i :
            if i+'_deformed_seg.nii.gz' not in os.listdir(fixed_masks):
                move_img = os.path.join(move_imgs, i)
                move_mask = os.path.join(move_masks, i)
                fixed_img = os.path.join(fixed_imgs, i)
                fixed_mask = os.path.join(fixed_masks, i)
                fixed = sitk.ReadImage(fixed_img)
                moving = sitk.ReadImage(move_img)
                print(fixed)
                #print(moving)


                if fixed.GetSize()[2]==moving.GetSize()[2]:
                    os.system(f'/data/deedsBCV/deedsBCV -F {fixed_img} -M {move_img}  -O {fixed_mask} -S {move_mask}')
                else:
                    img_r = resize_image_itk(fixed,moving, resamplemethod=sitk.sitkLinear)
                    sitk.WriteImage(img_r, fixed_img)
                    os.system(f'/data/deedsBCV/deedsBCV -F {fixed_img} -M {move_img}  -O {fixed_mask} -S {move_mask}')
        else:
            print(f'not found {i}')
        #except Exception as e:
        #    print('error',i,e)

            
move_imgs = sys.argv[1]
move_masks = sys.argv[2]

fixed_imgs = sys.argv[3]

fixed_masks = fixed_imgs + '_mask'
os.makedirs(fixed_masks, exist_ok=True)


run(move_imgs, move_masks, fixed_imgs, fixed_masks)
