import nibabel as nib
import numpy as np
from scipy.ndimage import label, find_objects
import os
import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes
from scipy.spatial import ConvexHull, Delaunay
import nibabel as nib
import numpy as np
from scipy.ndimage import label, binary_dilation
from skimage.measure import regionprops, label as sk_label
import glob

import nibabel as nib
import numpy as np
from scipy.spatial.distance import cdist

import nibabel as nib
import numpy as np
import csv


import nibabel as nib
import numpy as np
import cv2
from scipy.spatial import ConvexHull


import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
import cv2



def cal_ct(image_path,mask_path):

    image_nifti = nib.load(image_path)
    mask_nifti = nib.load(mask_path)

    image_data = image_nifti.get_fdata()
    mask_data = mask_nifti.get_fdata()

    # 获取标记为 label1 的区域的 CT 值
    label1_values = image_data[mask_data == 1]  # 假设 label1 的值为 1

    # 计算最大、最小和平均 CT 值
    max_ct_value = np.max(label1_values)
    min_ct_value = np.min(label1_values)
    mean_ct_value = np.mean(label1_values)

    return max_ct_value,min_ct_value,mean_ct_value



# 加载NIfTI文件
def load_nifti_file(file_path):
    nifti_img = nib.load(file_path)
    nifti_data = nifti_img.get_fdata()
    return nifti_data

# 检查是否有引流管（label 3）
def check_label_3_existence(mask_data):
    return np.any(mask_data == 1)

# 计算引流管（label 3）的个数
def count_label_3_objects(mask_data):
    labeled_array, num_features = label(mask_data == 1)
    return num_features

# 外扩指定像素的数量
def dilate_label(mask_data, label_value, dilation_size):
    label_mask = (mask_data == label_value)
    dilated_mask = binary_dilation(label_mask, iterations=dilation_size)
    return dilated_mask

# 计算label 3与label 2的接触边界直径
def calculate_contact_diameter(mask_data, dilation_size=1):
    # 外扩label 3
    dilated_label_3 = dilate_label(mask_data, 1, dilation_size)
    # 计算与label 2的接触区域
    contact_area = np.logical_and(dilated_label_3, mask_data == 2)
    # 计算接触区域的直径
    labeled_contact = sk_label(contact_area)
    properties = regionprops(labeled_contact)
    if properties:
        max_diameter = max(prop.major_axis_length for prop in properties)
    else:
        max_diameter = 0
    return max_diameter

# 计算label 3与label 2的接触面积
def calculate_contact_area(mask_data, dilation_size=1):
    # 外扩label 3
    dilated_label_3 = dilate_label(mask_data, 3, dilation_size)
    # 计算与label 2的接触区域
    contact_area = np.logical_and(dilated_label_3, mask_data == 2)
    # 计算接触区域的体素数量
    contact_area_size = np.sum(contact_area)
    return contact_area_size



import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import cdist

def load_label_data(file_path, label_value=1):
    # Load NIfTI file
    img = nib.load(file_path)
    data = img.get_fdata()
    # Extract the specified label
    label_data = (data == label_value).astype(np.uint8)
    return label_data, img.header.get_zooms()

def find_contact_area(label1, label2):
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import binary_dilation, generate_binary_structure
    import cv2

    # 加载NIfTI文件
    mask_img1 = nib.load(label1)
    mask_data1 = mask_img1.get_fdata()

    mask_img2 = nib.load(label2)
    mask_data2 = mask_img2.get_fdata()

    # 假设你的目标标签是1和2
    label1 = 1
    label2 = 1

    # 生成二值掩模
    binary_mask1 = (mask_data1 == label1).astype(np.uint8)
    binary_mask2 = (mask_data2 == label2).astype(np.uint8)

    # 获取体素尺寸（以毫米为单位）
    voxel_size = mask_img1.header.get_zooms()  # 返回 (x, y, z) 体素尺寸
    slice_thickness = voxel_size[2]  # z 方向的体素尺寸

    # 对label2进行外扩
    structure = generate_binary_structure(3, 5)  # 2D 膨胀
    dilated_mask2 = binary_dilation(binary_mask2, structure)

    # 计算重叠区域
    overlap = np.logical_and(binary_mask1, dilated_mask2)

    new_mask = np.zeros_like(overlap)  # 创建与原掩模相同形状的零数组
    new_mask[overlap] = 1  # 将交叉部分设置为1

    mask_data = new_mask
    total_max_diameter_mm=0
    # 假设你的目标标签是1
    label = 1
    binary_mask = (mask_data == label).astype(np.uint8)
    # 获取体素尺寸（以毫米为单位）
    voxel_size = mask_img2.header.get_zooms()  # 返回 (x, y, z) 体素尺寸
    slice_thickness = voxel_size[2]
    total_volume_mm3 =0
    # 计算每层的最大外接圆直径
    for i in range(binary_mask.shape[2]):  # 遍历z轴
        slice_mask = binary_mask[:, :, i]

        # 查找轮廓
        contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 合并所有轮廓
            all_contours = np.vstack(contours)
            
            # 拟合最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(all_contours)

            # 计算直径（直径 = 2 * 半径），并转换为毫米
            min_diameter_mm = 2 * radius * np.mean(voxel_size[:2])  # 使用 x 和 y 方向的平均体素尺寸=

            if min_diameter_mm>total_max_diameter_mm:
                total_max_diameter_mm=min_diameter_mm


            # 计算该层的面积（像素数 × 体素尺寸平方）
            area_mm2 = cv2.contourArea(all_contours) * (voxel_size[0] * voxel_size[1])
            total_volume_mm3 += area_mm2 * slice_thickness  # 增加体积

    return total_volume_mm3,total_max_diameter_mm

def calculate_area(contact_area, voxel_size):
    # Calculate the number of voxels in the contact area
    contact_voxels = np.sum(contact_area)
    # Convert to real-world area using voxel size
    voxel_volume = np.prod(voxel_size)
    contact_area_mm2 = contact_voxels * voxel_volume
    return contact_area_mm2

def calculate_max_diameter(contact_area, voxel_size):
    # Get the coordinates of contact voxels
    contact_points = np.argwhere(contact_area)
    if len(contact_points) < 2:
        return 0  # No sufficient points to calculate a diameter
    # Convert voxel indices to real-world coordinates
    real_world_coords = contact_points * voxel_size
    # Calculate all pairwise distances
    distances = cdist(real_world_coords, real_world_coords)
    # Find the maximum distance
    max_diameter = np.max(distances)
    return max_diameter



def calculate_fuqiang_bingzaometrics(image_file,nifti_file, label_value=1):
    # Load the NIfTI file
    nii = nib.load(nifti_file)
    image_data = nii.get_fdata()

    max_ct_value,min_ct_value,mean_ct_value = cal_ct(image_file,nifti_file)
    
    # Extract the ROI where the label matches label_value


    # 加载NIfTI文件
    mask_img = nib.load(nifti_file)
    mask_data = mask_img.get_fdata()

    total_max_diameter_mm=0

    # 假设你的目标标签是1
    label = 1
    binary_mask = (mask_data == label).astype(np.uint8)

    # 获取体素尺寸（以毫米为单位）
    voxel_size = mask_img.header.get_zooms()  # 返回 (x, y, z) 体素尺寸

    slice_thickness = voxel_size[2]

    total_volume_mm3 =0
    # 计算每层的最大外接圆直径
    for i in range(binary_mask.shape[2]):  # 遍历z轴
        slice_mask = binary_mask[:, :, i]

        # 查找轮廓
        contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 合并所有轮廓
            all_contours = np.vstack(contours)
            
            # 拟合最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(all_contours)

            # 计算直径（直径 = 2 * 半径），并转换为毫米
            min_diameter_mm = 2 * radius * np.mean(voxel_size[:2])  # 使用 x 和 y 方向的平均体素尺寸

            if min_diameter_mm>total_max_diameter_mm:
                total_max_diameter_mm=min_diameter_mm


            # 计算该层的面积（像素数 × 体素尺寸平方）
            area_mm2 = cv2.contourArea(all_contours) * (voxel_size[0] * voxel_size[1])
            total_volume_mm3 += area_mm2 * slice_thickness  # 增加体积

    # 计算在3D空间中的最小内接球直径
    # 获取所有非零点的坐标
    coordinates = np.argwhere(binary_mask > 0)

    # 计算最小包围球的直径
    if coordinates.size > 0:
        hull = ConvexHull(coordinates)
        hull_points = coordinates[hull.vertices]

        # 计算质心
        centroid = np.mean(hull_points, axis=0)

        # 计算每个点到质心的距离
        distances = np.linalg.norm(hull_points - centroid, axis=1)

        # 内接球的直径是最远点到质心的两倍
        min_radius = np.max(distances)
        min_diameter_mm = 2 * min_radius * np.mean(voxel_size)  # 使用 x, y 和 z 方向的平均体素尺寸

        print(f'Min Diameter in 3D Space = {min_diameter_mm:.2f} mm')



    return total_volume_mm3,min_diameter_mm,total_max_diameter_mm,max_ct_value,min_ct_value,mean_ct_value





import nibabel as nib
import numpy as np
from scipy.ndimage import label, find_objects, center_of_mass, distance_transform_edt

def calculate_bubble_metrics(image_file_path, nifti_file, label_value=1):

    # Load the NIfTI file
    nii_image = nib.load(image_file_path)
    image_data = nii_image.get_fdata()

    max_ct_value,min_ct_value,mean_ct_value = cal_ct(image_file_path,nifti_file)

    # Load the NIfTI file
    nii = nib.load(nifti_file)
    mask_data = nii.get_fdata()
    
    # Extract the ROI where the label matches label_value
    bubble_mask = (mask_data == label_value)
    
    # Voxel volume calculation (mm^3)
    voxel_volume = np.prod(nii.header.get_zooms())
    
    # Label connected components (bubbles)
    labeled_array, num_bubbles = label(bubble_mask)
    
    # Initialize variables for metrics
    total_volume = 0
    bubble_volumes = []
    bubble_densities = []
    
    # Calculate volume and density for each bubble
    for bubble_label in range(1, num_bubbles + 1):
        # Get the bubble mask
        bubble = (labeled_array == bubble_label)
        bubble_volume = np.sum(bubble) * voxel_volume
        bubble_density = mask_data[bubble]
        average_density = np.mean(bubble_density)
        max_density = np.max(bubble_density)
        min_density = np.min(bubble_density)
        
        # Store metrics
        total_volume += bubble_volume
        bubble_volumes.append(bubble_volume)
        bubble_densities.append((average_density, max_density, min_density))
    
    # Calculate metrics related to all bubbles
    if num_bubbles > 0:
        average_bubble_volume = np.mean(bubble_volumes)
        max_bubble_volume = np.max(bubble_volumes)
        min_bubble_volume = np.min(bubble_volumes)
    else:
        average_bubble_volume = max_bubble_volume = min_bubble_volume = 0
    
    # Calculate pairwise minimum distance between bubble centers of mass
    centers_of_mass = np.array(center_of_mass(bubble_mask, labeled_array, range(1, num_bubbles + 1)))
    if len(centers_of_mass) > 1:
        distances = np.linalg.norm(centers_of_mass[:, np.newaxis] - centers_of_mass[np.newaxis, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        min_distance_between_bubbles = np.min(distances)
    else:
        min_distance_between_bubbles = 0

    # Calculate the diameter of the minimum enclosing sphere for all bubbles
    if len(centers_of_mass) > 0:
        overall_center = np.mean(centers_of_mass, axis=0)
        overall_radius = np.max(np.linalg.norm(centers_of_mass - overall_center, axis=1))
        enclosing_sphere_diameter = 2 * overall_radius
        enclosing_sphere_volume = (4/3) * np.pi * overall_radius**3
    else:
        enclosing_sphere_diameter = enclosing_sphere_volume = 0
    
    # Volume ratio (total bubble volume / enclosing sphere volume)
    volume_ratio = total_volume / enclosing_sphere_volume if enclosing_sphere_volume > 0 else 0

    return total_volume,num_bubbles,average_bubble_volume,max_bubble_volume,min_bubble_volume,min_distance_between_bubbles,enclosing_sphere_diameter,volume_ratio,max_ct_value,min_ct_value,mean_ct_value


def calculate_metrics_ulg(mask_file_path_bingzao,mask_file_path_tube):
    #mask_data_tube = load_nifti_file(mask_file_path_tube)
    mask_img_tube = nib.load(mask_file_path_tube)
    mask_data_tube = mask_img_tube.get_fdata()

    label1_data, voxel_size1 = load_label_data(mask_file_path_bingzao)
    label2_data, voxel_size2 = load_label_data(mask_file_path_tube)

    # Ensure both NIfTI files have the same voxel size
    if voxel_size1 != voxel_size2:
        raise ValueError("The voxel sizes of the two NIfTI files do not match.")

    nii_label2 = nib.load(mask_file_path_bingzao)
    data_label2 = nii_label2.get_fdata()

    nii_label1 = nib.load(mask_file_path_tube)
    data_label1 = nii_label1.get_fdata()

    max_total_contact_area,max_total_contact_perimeter=0,0

    if len(np.unique(data_label1))==1 or len(np.unique(data_label2))==1:
        total_volume_mm3,total_max_diameter_mm=0,0
    else:
        # Find the contact area between the two labels
        total_volume_mm3,total_max_diameter_mm = find_contact_area(mask_file_path_bingzao, mask_file_path_tube)

    return total_volume_mm3,total_max_diameter_mm



import numpy as np
import nibabel as nib
from scipy.ndimage import label
from scipy.spatial import ConvexHull

# 加载NIfTI文件
def load_nifti_file(file_path):
    nifti_img = nib.load(file_path)
    nifti_data = nifti_img.get_fdata()
    return nifti_data, nifti_img.affine

# 找到所有气泡的坐标
def get_bubble_coordinates(mask_data, label_value):
    # 获取所有指定label的体素坐标
    coordinates = np.argwhere(mask_data == label_value)
    return coordinates

# 计算最小外接立方体的范围
def calculate_bounding_box(coordinates):
    min_coords = np.min(coordinates, axis=0)
    max_coords = np.max(coordinates, axis=0)
    return min_coords, max_coords

# 计算凸包（可选）
def calculate_convex_hull(coordinates):
    if coordinates.shape[0] < 4:  # 至少需要四个点来构建三维凸包
        return None
    hull = ConvexHull(coordinates)
    return hull

# 生成包含所有气泡的最小区域的掩膜
def create_bounding_box_mask(mask_shape, min_coords, max_coords):
    bounding_box_mask = np.zeros(mask_shape, dtype=np.uint8)
    bounding_box_mask[
        min_coords[0]:max_coords[0]+1,
        min_coords[1]:max_coords[1]+1,
        min_coords[2]:max_coords[2]+1
    ] = 1
    return bounding_box_mask

def save_nifti_file(data, affine, output_path):
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)



def load_nifti_file(file_path):
    nifti_img = nib.load(file_path)
    nifti_data = nifti_img.get_fdata()
    return nifti_data, nifti_img.affine

# 保存NIfTI文件
def save_nifti_file(data, affine, output_path):
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)

# 找到所有气泡床（label 4）的坐标
def get_bubble_bed_coordinates(mask_data, label_value):
    coordinates = np.argwhere(mask_data == label_value)
    return coordinates

# 构建凸包，并生成气泡床的mask
def create_bubble_bed_mask(mask_shape, coordinates):
    if coordinates.shape[0] < 3:
        return None  # 如果点数少于3，无法生成三维的凸包
    hull = ConvexHull(coordinates)
    # 创建一个新的mask，初始化为0
    bubble_bed_mask = np.zeros(mask_shape, dtype=np.uint8)
    # 设置凸包区域内的点为1
    for simplex in hull.simplices:
        pts = coordinates[simplex]
        bubble_bed_mask[pts[:, 0], pts[:, 1], pts[:, 2]] = 1
    return bubble_bed_mask



def create_big_bed(mask_file_path,outputpath,mask_file_path_bingzao):


    import numpy as np
    import nibabel as nib
    from scipy.ndimage import binary_dilation

    # 加载病灶和气泡的 NIfTI 文件
    lesion_img = nib.load(mask_file_path_bingzao)
    bubbles_img = nib.load(mask_file_path)

    # 获取数据和仿射变换矩阵
    lesion_data = lesion_img.get_fdata()
    bubbles_data = bubbles_img.get_fdata()
    affine = lesion_img.affine

    # 创建一个膨胀结构元素，使用3D立方体
    structure = np.ones((3, 3, 3), dtype=np.uint8)

    # 对气泡的 mask 进行膨胀操作，迭代10次
    dilated_bubbles_mask = binary_dilation(bubbles_data > 0, structure=structure, iterations=10).astype(np.uint8)

    # 合并膨胀后的气泡 mask 和病灶的 mask，非零值位置设置为1
    combined_mask = np.logical_or(dilated_bubbles_mask > 0, lesion_data > 0).astype(np.uint8)


    nii_file = mask_file_path
    img = nib.load(nii_file)
    data = img.get_fdata()
    if np.any(data):

        # 2. 合并多个气泡的mask
        # 将所有非零值转换为1来合并成一个整体的二值化mask
        merged_mask = (data > 0).astype(np.uint8)

        # 3. 提取所有气泡的边界点
        # 使用marching_cubes提取表面网格，得到边界的顶点坐标
        verts, faces, _, _ = marching_cubes(merged_mask, level=0)

        # 4. 生成包含所有气泡的最小凸包
        # 创建ConvexHull来生成包含所有边界点的凸包
        hull = ConvexHull(verts)

        # 5. 生成新的多边形mask
        # 使用Delaunay三角剖分将凸包体素映射回3D体数据
        grid_x, grid_y, grid_z = np.indices(merged_mask.shape)
        points = np.c_[grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]
        del_tri = Delaunay(verts[hull.vertices])

        # 判断每个点是否在凸包内，如果在内部，则设为1
        mask_convex_hull = del_tri.find_simplex(points) >= 0
        mask_convex_hull = mask_convex_hull.reshape(merged_mask.shape).astype(np.uint8)

    overlap_mask = np.logical_and(combined_mask > 0, mask_convex_hull > 0).astype(np.uint8)

    # 将重叠的 mask 保存为新的 NIfTI 文件
    overlap_img = nib.Nifti1Image(overlap_mask, affine)
    nib.save(overlap_img, outputpath+'/'+mask_file_path.split('/')[-1])




def load_nifti(file_path):
    """Load NIfTI file and return the data array and affine."""
    nii = nib.load(file_path)
    data = nii.get_fdata()
    affine = nii.affine
    return data, affine

def calculate_volume(data, voxel_volume):
    """Calculate the volume of the labeled region."""
    return np.sum(data > 0) * voxel_volume

def calculate_max_diameter(data, voxel_size):
    """Calculate the maximum diameter of the labeled region."""
    coords = np.array(np.where(data > 0)).T * voxel_size
    max_distance = np.max(cdist(coords, coords))
    return max_distance

def calculate_density_statistics(image_data,mask_data,label=1):
    # Get the values from image_data where mask_data equals the specified label
    label_values = image_data[mask_data == label]

    if label_values.size == 0:
        return None, None, None  # Return None if no values found for the label

    # Calculate statistics
    mean_density = np.mean(label_values)
    max_density_95 = np.max(label_values)
    min_density_5 = np.min(label_values)
    return mean_density, max_density_95, min_density_5

def calculate_volume_ratio(data, label1, label2, voxel_volume):
    """Calculate the volume ratio of two labels."""
    volume_label1 = np.sum(data == label1) * voxel_volume
    volume_label2 = np.sum(data == label2) * voxel_volume
    return volume_label1 / volume_label2 if volume_label2 > 0 else None

def main_qipao_bed(image_file,nifti_file, voxel_size):
    data, affine = load_nifti(nifti_file)
    voxel_volume = np.prod(voxel_size)
    # Calculate volume



    img = nib.load(nifti_file)
    data = img.get_fdata()
    voxel_sizes = np.abs(img.header.get_zooms())
    voxel_volume = np.prod(voxel_sizes)  # 每个体素的体积（立方毫米）
    # 计算非零体素的数量
    non_zero_voxels = np.count_nonzero(data)
    # 计算总体积（立方毫米）
    volume = non_zero_voxels * voxel_volume

    # Calculate maximum diameter

    binary_mask = (data == 1).astype(np.uint8)
    # 获取体素尺寸（以毫米为单位）
    voxel_size = img.header.get_zooms()  # 返回 (x, y, z) 体素尺寸
    slice_thickness = voxel_size[2]
    total_max_diameter_mm=0
    # 计算每层的最大外接圆直径
    for i in range(binary_mask.shape[2]):  # 遍历z轴
        slice_mask = binary_mask[:, :, i]

        # 查找轮廓
        contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 合并所有轮廓
            all_contours = np.vstack(contours)
            
            # 拟合最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(all_contours)

            # 计算直径（直径 = 2 * 半径），并转换为毫米
            min_diameter_mm = 2 * radius * np.mean(voxel_size[:2])  # 使用 x 和 y 方向的平均体素尺寸
            if min_diameter_mm>total_max_diameter_mm:
                total_max_diameter_mm=min_diameter_mm



    max_diameter = total_max_diameter_mm
    # Calculate density statistics
    image_data = load_nifti(image_file)
    mask_data = data
    max_ct_value,min_ct_value,mean_ct_value = cal_ct(image_file,nifti_file)
    return volume,max_diameter,max_ct_value,min_ct_value,mean_ct_value

def run2(nifti_file_path_mask_,image_path,label1_PATH):
    label1_data=[]
    try:    
        print(nifti_file_path_mask_)
        image_file_path = os.path.join(image_path+'/'+nifti_file_path_mask_.split('/')[-1])
        nii_label = nib.load(nifti_file_path_mask_)
        data_label = nii_label.get_fdata()
        if len(np.unique(data_label))!=1:
            total_volume,max_diameter_3d,max_diameter_2d,max_ct_value,min_ct_value,mean_ct_value= calculate_fuqiang_bingzaometrics(image_file_path,nifti_file_path_mask_, label_value=1)
            print(total_volume,max_diameter_3d,max_diameter_2d,max_ct_value,min_ct_value,mean_ct_value)
            label1_data.append([nifti_file_path_mask_.split('/')[-1],total_volume,max_diameter_3d,max_diameter_2d,max_ct_value,min_ct_value,mean_ct_value])
    except Exception as e:
        print(e)

    csv_filename = 'portal_task1-label1/'+nifti_file_path_mask_.split('/')[-1]+'.csv'
    # 写入 CSV 文件
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id','total_volume','max_diameter_3d','max_diameter_2d','max_ct_value','min_ct_value','mean_ct_value'])  # 写入表头
        writer.writerows(label1_data)  # 写入数据





# Example usage
#nifti_file_path = r'mask_bubble_preprocess_preprocess'
image_path = r'portal_image_preprocess'
#nifti_file_path_mask_fluid = r'mask_fluid_preprocess_preprocess'
#os.makedirs(outputpath, exist_ok=True)

choose='3'
from  multiprocessing import Process,Pool
total_data=[]
label1_data=[]

"""
label1_PATH = r'portal_mask_fix_preprocess_1_1'
pool = Pool(30)
for nifti_file_path_mask_ in glob.glob(label1_PATH+"/*.nii.gz"):
    print(nifti_file_path_mask_)
    pool.apply_async(func=run2, args=(nifti_file_path_mask_,image_path,label1_PATH,))
pool.close()
pool.join()



"""
"""

def run1(image_path,nifti_file_path_mask_,label1_PATH,label2_PATH):
    label2_data=[]
    label2_2_data=[]
    try: 
        image_file_path = os.path.join(image_path+'/'+nifti_file_path_mask_.split('/')[-1])
        nii_label = nib.load(nifti_file_path_mask_)
        data_label = nii_label.get_fdata()
        if len(np.unique(data_label))!=1:
            total_volume,max_diameter_3d,max_diameter_2d,max_ct_value,min_ct_value,mean_ct_value= calculate_fuqiang_bingzaometrics(image_file_path,nifti_file_path_mask_, label_value=1)
            print(total_volume,max_diameter_3d,max_diameter_2d,max_ct_value,min_ct_value,mean_ct_value)
            label2_data.append([nifti_file_path_mask_.split('/')[-1],total_volume,max_diameter_3d,max_diameter_2d,max_ct_value,min_ct_value,mean_ct_value])

            label1_path=os.path.join(label1_PATH,nifti_file_path_mask_.split('/')[-1])
            print(label1_path)
            if os.path.exists(label1_path):
                total_volume_mm3,total_max_diameter_mm =calculate_metrics_ulg(label1_path,nifti_file_path_mask_)
                print(total_volume_mm3,total_max_diameter_mm)
                label2_2_data.append([nifti_file_path_mask_.split('/')[-1],total_volume_mm3,total_max_diameter_mm])
    except Exception as e:
        print(e)

    csv_filename = 'portal_task1-label2/'+nifti_file_path_mask_.split('/')[-1]+'.csv'
    # 写入 CSV 文件
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id','total_volume','max_diameter_3d','max_diameter_2d','max_ct_value','min_ct_value','mean_ct_value'])  # 写入表头
        writer.writerows(label2_data)  # 写入数据

    csv_filename = 'portal_task1-label-1-2/'+nifti_file_path_mask_.split('/')[-1]+'.csv'
    # 写入 CSV 文件
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id','total_volume_mm3','total_max_diameter_mm'])  # 写入表头
        writer.writerows(label2_2_data)  # 写入数据





label1_PATH = r'portal_mask_fix_preprocess_1_1'

pool = Pool(30)
label2_PATH = r'portal_mask_fix_preprocess_2_1'
for nifti_file_path_mask_ in glob.glob(label2_PATH+"/*.nii.gz"):
    print(nifti_file_path_mask_)
    pool.apply_async(func=run1, args=(image_path,nifti_file_path_mask_,label1_PATH,label2_PATH,))
pool.close()
pool.join()

"""

import pandas as pd
path2="portal_task1-label-1-2"

csv_files = [os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.csv')]
# 使用pandas读取并合并所有CSV文件
df_list = [pd.read_csv(file) for file in csv_files]
combined_df2 = pd.concat(df_list, ignore_index=True)
combined_df2.to_csv(path2+'.csv', index=False) 



path2="portal_task1-label1"

csv_files = [os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.csv')]
# 使用pandas读取并合并所有CSV文件
df_list = [pd.read_csv(file) for file in csv_files]
combined_df2 = pd.concat(df_list, ignore_index=True)
combined_df2.to_csv(path2+'.csv', index=False) 


path2="portal_task1-label2"

csv_files = [os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.csv')]
# 使用pandas读取并合并所有CSV文件
df_list = [pd.read_csv(file) for file in csv_files]
combined_df2 = pd.concat(df_list, ignore_index=True)
combined_df2.to_csv(path2+'.csv', index=False) 
