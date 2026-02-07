import warnings
warnings.filterwarnings('ignore')
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
import os
import SimpleITK as sitk
from radiomics import featureextractor
import numpy as np
import pandas as pd
import datetime
import radiomics
import logging
logger = radiomics.logging.getLogger("radiomics")
logger.setLevel(radiomics.logging.ERROR)
import sys
import nibabel as nib
import numpy as np
def task(image_path,mask_path,eachlabel,eachfile,extractor,expand_name,output_path):
    integrated_result = extractor.execute(image_path,mask_path,label=eachlabel)
    df_new = pd.DataFrame.from_dict(integrated_result, orient='index').T
    file_name_without_ext = eachfile.replace(".nii.gz", "")  # 去掉扩展名
    df_new.columns=['liuzhou_3d_exp_label_'+str(eachlabel)+'_'+expand_name+'_'+col for col in df_new.columns]
    df_new.insert(0, "ID", file_name_without_ext)  # 在第一列插入文件名
    columns_keeps = [col for col in df_new.columns if 'diagnostics' not in col]
    df_new = df_new[columns_keeps]
    print(output_path+'/'+eachfile+'_liuzhou_3d.csv')
    df_new.to_csv(output_path+'/'+eachfile+'_liuzhou_3d.csv', index=False) 


def run(path1,path2,eachfile,output_path,output_path_csv,extractor):
    try:   
        print(datetime.datetime.now(),eachfile)
        #for eachfile in os.listdir(path1):
        image_path=os.path.join(path1,eachfile)
        mask_path=os.path.join(path2,eachfile)

        expand_name=path2.split('/')[-1]

        #try:
        print(image_path)
        print(mask_path)
        
        #pool2 = Pool(10)
        each_feature=[]
        
        nii_img = nib.load(mask_path)
        data = nii_img.get_fdata()
        exists = np.any(data == 2)
        
        if exists:
            for eachlabel in [2]:
                task(image_path,mask_path,eachlabel,eachfile,extractor,expand_name,output_path)
                #pool2.apply_async(func=task, args=(image_path,mask_path,eachlabel,eachfile,extractor,expand_name,output_path,))
        #pool2.close()
        #pool2.join()

    except Exception as e:
        print(e)
    
    

    #except Exception as e:
    #    print(e)
    #    tmp = pd.DataFrame()
    #    return tmp


if __name__ == '__main__':
    import sys
    path1=sys.argv[1]
    path2=sys.argv[2]
    output_path= path2+'_csv'
    output_path_csv= output_path+'.csv'
    os.makedirs(output_path, exist_ok=True)
    

    finalresult=[]
    from  multiprocessing import Process,Pool
    pool2 = Pool(10)
    df1 = pd.DataFrame()
    print(output_path)
    doneid=[i.split('.')[0] for i in os.listdir(output_path)]
    
    extractor = featureextractor.RadiomicsFeatureExtractor(geometryTolerance = 9999)
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures() # 使用所有特征
    
    for eachfile in os.listdir(path1)[::-1]:
        if eachfile not in os.listdir(path2):
            continue
        #try:    
        if eachfile.split('.')[0] in doneid or eachfile.split('.')[0][1:] in doneid or '0'+eachfile.split('.')[0] in doneid:
            pass
        else:
            print('no')   
            print(eachfile)
            #I =run(path1,path2,eachfile)
            #print(I)
            #finalresult.append(pool1.apply_async(func=run, args=(path1,path2,eachfile,)))
            #run(path1,path2,eachfile,output_path,output_path_csv,extractor)
            pool2.apply_async(func=run, args=(path1,path2,eachfile,output_path,output_path_csv,extractor,))
    pool2.close()
    pool2.join()

    
    csv_files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith('.csv')]
    df_list = [pd.read_csv(file, dtype={0: str}) for file in csv_files]
    combined_df1 = pd.concat(df_list, ignore_index=True)
    combined_df1.to_csv(output_path_csv, index=False)



