import pandas as pd





import pandas as pd

import pandas as pd
"""
# 读取 CSV 文件
df = pd.read_csv(r'mask_fix_preprocess_3_1_feature.csv')
print(df)

df = pd.read_csv(r'mask_fix_preprocess_5_1_feature.csv')

# 筛选出非数值的列，排除ID列
non_numeric_columns = [col for col in df.columns if col != 'ID' and not pd.api.types.is_numeric_dtype(df[col])]

# 删除这些列
df.drop(columns=non_numeric_columns, inplace=True)
print(df)
df.to_csv('mask_fix_preprocess_5_1_feature_new.csv', index=False)

"""
c=pd.read_csv(r"portal_mask_1_1_feature_csv_filter_feature_name.csv")


c_list=list(c.iloc[:,0])
print(c_list)

c_data=pd.read_csv(r"portal_mask_1_1_feature_csv.csv")

filter_col_c =[col for col in c_list if col in c_data.columns]

new_col_c=[]

table1 = c_data[['ID']+filter_col_c]

table1.to_csv('task1_portal_mask_1_1_filter_feature_csv.csv', index=False)