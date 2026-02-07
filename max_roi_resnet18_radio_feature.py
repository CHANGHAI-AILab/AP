import warnings
warnings.filterwarnings('ignore')
import os
import SimpleITK as sitk
from radiomics import featureextractor
import numpy as np
import pandas as pd
import datetime
import yaml
import radiomics
import logging
from  multiprocessing import Process,Pool
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import torch.nn as nn

logger = radiomics.logging.getLogger("radiomics")
logger.setLevel(radiomics.logging.ERROR)

def find_mask(slice_mask,slice_image,extractor):
    if 1 not in np.unique(slice_mask):
        return None
    else:
        count_piex=sum(slice_mask)
        if count_piex<=1:
            return None
        else:
            try:
                features_per_s=extractor.execute(slice_image, slice_mask)
                return features_per_s
            except Exception as e:
                print(e)
                return None


def run_2d(image_folder,model,image_file,output_path):
    image_filenames = []
    feature_vectors = []
    image_path = os.path.join(image_folder, image_file)
    features = extract_features(image_path, model)
    image_filenames.append(image_file)
    feature_vectors.append(features)
    # Convert to a DataFrame (each row is an image, each column a feature)
    df = pd.DataFrame(feature_vectors)
    df.columns = [f'resnet18_{i+1}' for i in range(df.shape[1])]
    df.insert(0, 'ID', image_filenames)  # Insert image filenames as the first column
    # Save to CSV
    df.to_csv(output_path+'/'+image_file+'_resnet18.csv', index=False) 
    print("Features extracted and saved to resnet18_features.csv")



def run_3d(path1,path2,eachfile,output_path):

    print(datetime.datetime.now(),eachfile)
    #for eachfile in os.listdir(path1):
    image_path=os.path.join(path1,eachfile)
    mask_path=os.path.join(path2,eachfile)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures() # 使用所有特征

    print('start get 3d feature')

    integrated_result = extractor.execute(image_path, mask_path)
    print('start finish 3d feature',len(integrated_result))
    df_new = pd.DataFrame.from_dict(integrated_result, orient='index').T
    file_name_without_ext = eachfile.replace(".nii.gz", "")  # 去掉扩展名
    df_new.columns=['3d_'+col for col in df_new.columns]
    df_new.insert(0, "ID", file_name_without_ext)  # 在第一列插入文件名
    columns_keeps = [col for col in df_new.columns if 'diagnostics' not in col]
    df_new = df_new[columns_keeps]
    print(output_path+'/'+eachfile+'_3d.csv')
    df_new.to_csv(output_path+'/'+eachfile+'_3d.csv', index=False) 

# Function to extract features from a single image
def extract_features(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():  # Disable gradient calculation
        features = model(image.cuda())
    return features.squeeze().cpu().numpy()  # Remove batch dimension

if __name__ == '__main__':
    import sys
    path1=sys.argv[1]
    path2=path1+'_resnet18_csv'
    output_path=path2+'.csv'
    chooese = sys.argv[2]

    #2d max roi dir
    #save each csv dir
    #to csv name

    #3d image dir
    #mask dir 
    #csv dir




    if chooese=='2d':
        os.makedirs(path2, exist_ok=True) 
        # Initialize ResNet50 model (pre-trained)
        model = models.resnet18(pretrained=True).cuda()
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()  # Set model to evaluation mode

        # Define transformation (ResNet50 expects 224x224 input images)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Folder containing the images
        image_folder = path1
        # Prepare list to store feature vectors
        image_filenames = []
        feature_vectors = []

        # Loop through the image folder and extract features for each image
        for image_file in os.listdir(image_folder):
            print(image_file)
            #if image_file.endswith(".jpg"):
            image_path = os.path.join(image_folder, image_file)
            run_2d(image_folder,model,image_file,path2)

        csv_files = [os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.csv')]
        # 使用pandas读取并合并所有CSV文件
        df_list = [pd.read_csv(file) for file in csv_files]
        combined_df1 = pd.concat(df_list, ignore_index=True)
        combined_df1.to_csv(output_path, index=False) 

    if chooese=='3d':

        pool2 = Pool(15)
        df1 = pd.DataFrame()
        finalresult=[]
        from  multiprocessing import Process,Pool
        #datacvs=pd.read_csv(csv_path_sys)
        os.makedirs(output_path, exist_ok=True) 

        doneid=[]
        for each in os.listdir(output_path):
            doneid.append(str(each.split('/')[-1].split('.nii.gz')[0]).strip())

        for eachfile in os.listdir(path1):
            try:    
                if eachfile.split('.')[0] in doneid or eachfile.split('.')[0][1:] in doneid or '0'+eachfile.split('.')[0] in doneid:
                    print('already_done')
                else:
                    print('no')   
                    print(eachfile)
                    #run_3d(path1,path2,eachfile,output_path)
                    pool2.apply_async(func=run_3d, args=(path1,path2,eachfile,output_path,))
            except Exception as e:
                print(e,eachfile,'error')
        pool2.close()
        pool2.join()

        csv_files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith('.csv')]
        # 使用pandas读取并合并所有CSV文件
        df_list = [pd.read_csv(file) for file in csv_files]
        combined_df1 = pd.concat(df_list, ignore_index=True)
        combined_df1.to_csv(output_path+'.csv', index=False)

    if chooese=='2d3d':
        csv_files = [os.path.join(path1, f) for f in os.listdir(path1) if f.endswith('.csv')]
        # 使用pandas读取并合并所有CSV文件
        df_list = [pd.read_csv(file) for file in csv_files]
        combined_df1 = pd.concat(df_list, ignore_index=True)


        csv_files = [os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.csv')]
        # 使用pandas读取并合并所有CSV文件
        df_list = [pd.read_csv(file) for file in csv_files]
        combined_df2 = pd.concat(df_list, ignore_index=True)
        merged_table = combined_df1.merge(combined_df2, on='ID')
        merged_table.to_csv(output_path+'.csv', index=False) 
        #except Exception as e:
        #    print(e)