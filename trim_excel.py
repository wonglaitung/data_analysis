import pandas as pd
import os

# 定义数据目录和输出目录
data_dir = '/data/data_analysis/data_train'
output_dir = '/data/data_analysis/data_train'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历数据目录下的所有Excel文件
for filename in os.listdir(data_dir):
    if filename.endswith('.xlsx'):
        # 构造完整的文件路径
        file_path = os.path.join(data_dir, filename)
        
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 保留前100条记录
        df_trimmed = df.head(100)
        
        # 构造输出文件路径
        output_file_path = os.path.join(output_dir, filename)
        
        # 保存处理后的数据到新的Excel文件
        df_trimmed.to_excel(output_file_path, index=False)
        
        print(f'处理完成: {filename}')