import pandas as pd
import numpy as np
import shutil

# 读取ml_wide_table.csv文件
df = pd.read_csv('output/ml_wide_table.csv')

# 在Id后面增加Label字段，前1000个为1，后面所有都为0
n_samples = len(df)
# 前1000个为1，其余为0
n_ones = min(1000, n_samples)  # 防止数据总数少于1000的情况
n_zeros = n_samples - n_ones

# 创建标签数组：前1000个为1，后面为0
labels = np.concatenate([np.ones(n_ones), np.zeros(n_zeros)])

df.insert(1, 'Label', labels.astype(int))

# 保存到data/train.csv
df.to_csv('data/train.csv', index=False)

# 复制前100条数据到test.csv，并删除Label字段
test_df = df.head(100).drop('Label', axis=1)
test_df.to_csv('data/test.csv', index=False)

print("✅ 已将ml_wide_table.csv复制到data/train.csv，并添加了Label字段")
print("✅ 已复制前100条数据到data/test.csv，并删除了Label字段")