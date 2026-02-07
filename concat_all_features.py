import pandas as pd





#df1 = pd.read_csv("/data5/syx_feature/a_feature/a_features_timchen.csv")
#df2 = pd.read_csv("/data5/xb_feature/generalRCCPSQ_features_timchen.csv")
#print(set(df2.columns)-set(df1.columns))
"""
resnet18_path1="IPMN_t1_zengqiang_mask_fix_preprocess_max_roi_resnet18_csv.csv"
dinov2_path1="IPMN_t1_zengqiang_mask_fix_preprocess_max_roi_dinov2.csv"
table3 = pd.read_csv("IPMN_t1_zengqiang_mask_fix_preprocess_csv.csv")
table4 = pd.read_csv("IPMN_t1_zengqiang_mask_fix_preprocess_p5_csv.csv")
table5 = pd.read_csv("IPMN_t1_zengqiang_mask_fix_preprocess_p2_csv.csv")
table6 = pd.read_csv("IPMN_t1_zengqiang_image_preprocess_rad_features_c3.csv")
table7 = pd.read_csv("IPMN_t1_zengqiang_image_preprocess_rad_features_c5.csv")
outputname="IPMN_t1_zengqiang_demo_preprocess_feature.csv"

resnet18_path1="IPMN_t2_mask_fix_preprocess_max_roi_resnet18_csv.csv"
dinov2_path1="IPMN_t2_mask_fix_preprocess_max_roi_dinov2.csv"
table3 = pd.read_csv("IPMN_t2_mask_fix_preprocess_csv.csv")
table4 = pd.read_csv("IPMN_t2_mask_fix_preprocess_p5_csv.csv")
table5 = pd.read_csv("IPMN_t2_mask_fix_preprocess_p2_csv.csv")
table6 = pd.read_csv("IPMN_t2_image_preprocess_rad_features_c3.csv")
table7 = pd.read_csv("IPMN_t2_image_preprocess_rad_features_c5.csv")
outputname="IPMN_t2_demo_preprocess_feature.csv"



resnet18_path1="MCN_t1_zengqiang_mask_fix_preprocess_max_roi_resnet18_csv.csv"
dinov2_path1="MCN_t1_zengqiang_mask_fix_preprocess_max_roi_dinov2.csv"
table3 = pd.read_csv("MCN_t1_zengqiang_mask_fix_preprocess_csv.csv")
table4 = pd.read_csv("MCN_t1_zengqiang_mask_fix_preprocess_p5_csv.csv")
table5 = pd.read_csv("MCN_t1_zengqiang_mask_fix_preprocess_p2_csv.csv")
table6 = pd.read_csv("MCN_t1_zengqiang_original_preprocess_rad_features_c3.csv")
table7 = pd.read_csv("MCN_t1_zengqiang_original_preprocess_rad_features_c5.csv")
outputname="MCN_t1_zengqiang_demo_preprocess_feature.csv"


resnet18_path1="MCN_t2_mask_fix_preprocess_max_roi_resnet18_csv.csv"
dinov2_path1="MCN_t2_mask_fix_preprocess_max_roi_dinov2.csv"
table3 = pd.read_csv("MCN_t2_mask_fix_preprocess_csv.csv")
table4 = pd.read_csv("MCN_t2_mask_fix_preprocess_p5_csv.csv")
table5 = pd.read_csv("MCN_t2_mask_fix_preprocess_p2_csv.csv")
table6 = pd.read_csv("MCN_t2_original_preprocess_rad_features_c3.csv")
table7 = pd.read_csv("MCN_t2_original_preprocess_rad_features_c5.csv")
outputname="MCN_t2_demo_preprocess_feature.csv"
"""

resnet18_path1="portal_mask_fix_preprocess_1_1_max_roi_resnet18_csv.csv"
dinov2_path1="portal_mask_fix_preprocess_1_1_max_roi_dinov2.csv"
table3 = pd.read_csv("portal_mask_fix_preprocess_1_1_csv.csv")
table4 = pd.read_csv("portal_mask_fix_preprocess_1_1_p5_csv.csv")
table5 = pd.read_csv("portal_mask_fix_preprocess_1_1_p2_csv.csv")
outputname="portal_mask_1_1_feature_csv.csv"



resnet18_path1concat=resnet18_path1+"_concat.csv"
dinov2_path1concat=dinov2_path1+"_concat.csv"



# 示例数据

# 创建 DataFrame
df = pd.read_csv(resnet18_path1)
# 提取 ID 的前缀（.nii 之前的部分）和后缀（.nii 之后的部分）
df['base_ID'] = df['ID'].str.split('.nii').str[0]  # 提取 .nii 之前的部分
df['suffix'] = df['ID'].str.split('.nii').str[1]   # 提取 .nii 之后的部分

# 将原来的 ID 列去掉
df = df.drop(columns=['ID'])

# 对相同的 base_ID 进行合并
merged_df = df.pivot_table(index='base_ID', 
                           columns='suffix', 
                           values=['resnet18_'+str(i) for i in range(1,513)], 
                           aggfunc='first')

# 展平列索引并赋予新的列名
merged_df.columns = [f'{feat}_{suffix}' for feat, suffix in merged_df.columns]
# 重置索引
merged_df = merged_df.reset_index()
# 展示结果

merged_df.to_csv(resnet18_path1concat, index=False)



df = pd.read_csv(dinov2_path1)
# 提取 ID 的前缀（.nii 之前的部分）和后缀（.nii 之后的部分）
df['base_ID'] = df['ID'].str.split('.nii').str[0]  # 提取 .nii 之前的部分
df['suffix'] = df['ID'].str.split('.nii').str[1]   # 提取 .nii 之后的部分

# 将原来的 ID 列去掉
df = df.drop(columns=['ID'])

# 对相同的 base_ID 进行合并
merged_df = df.pivot_table(index='base_ID', 
                           columns='suffix', 
                           values=['big_model_feature'+str(i) for i in range(1,1025)], 
                           aggfunc='first')

# 展平列索引并赋予新的列名
merged_df.columns = [f'{feat}_{suffix}' for feat, suffix in merged_df.columns]
# 重置索引
merged_df = merged_df.reset_index()
# 展示结果

merged_df.to_csv(dinov2_path1concat, index=False)







table1 = pd.read_csv(resnet18_path1concat)
table2 = pd.read_csv(dinov2_path1concat)


print(table1.shape)
print(table2.shape)
print(table3.shape)
print(table4.shape)
print(table5.shape)


table1 = table1.rename(columns={'base_ID': 'ID'})
table2 = table2.rename(columns={'base_ID': 'ID'})
table3 = table3.rename(columns={'base_ID': 'ID'})
merged_table = table1.merge(table2, on='ID')
merged_table = merged_table.merge(table3, on='ID')
merged_table = merged_table.merge(table4, on='ID')
merged_table = merged_table.merge(table5, on='ID')

print(merged_table)
num_columns = merged_table.select_dtypes(include=['float64']).columns
columns_keep=['ID']+list(num_columns)
df_filter=merged_table[columns_keep]
print(df_filter)
df_filter.to_csv(outputname, index=False)
"""
table1 = pd.read_csv("mask_fix_preprocess_1_1_feature.csv")
print(table1)
table1.columns = ['ID']+[feat+'_label1' for feat in table1.columns[1:]]

table2 = pd.read_csv("mask_fix_preprocess_2_1_feature.csv")

table2.columns = ['ID']+[feat+'_label2' for feat in table2.columns[1:]]
print(table2)
merged_table = table1.merge(table2, on='ID',how='left')
print(merged_table)
merged_table.to_csv('task2_feature.csv', index=False)
"""
















