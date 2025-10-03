import pandas as pd
import sys
import os

def add_label_to_wide_table(excel_file_path, sheet_name='2024Raw', label_column='本地支薪'):
    """
    将标签添加到大宽表
    
    参数:
    excel_file_path: Excel文件路径
    sheet_name: 工作表名称，默认为'2024Raw'
    label_column: 用作标签的列名，默认为'本地支薪'
    """
    # 读取大宽表
    print("正在读取大宽表...")
    df_wide = pd.read_csv('./output/ml_wide_table_global.csv')
    
    # 读取Excel文件中的指定工作表
    print(f"正在读取Excel文件: {excel_file_path}, 工作表: {sheet_name}")
    df_excel = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    # 提取需要的列并去重
    print(f"正在处理标签数据，使用列: CI No. 和 {label_column}")
    df_label = df_excel[['CI No.', label_column]].drop_duplicates()
    df_label.rename(columns={'CI No.': 'Id', label_column: 'Label'}, inplace=True)
    
    # 处理ID列，确保数据类型正确并处理前导零问题
    # 将ID转换为字符串，去除前导零，再转换为整数
    df_label['Id'] = df_label['Id'].astype(str).str.replace('\\.0$', '', regex=True).str.lstrip('0')
    df_wide['Id'] = df_wide['Id'].astype(str).str.replace('\\.0$', '', regex=True).str.lstrip('0')
    
    # 转换为整数类型
    df_label['Id'] = pd.to_numeric(df_label['Id'], errors='coerce').astype('Int64')
    df_wide['Id'] = pd.to_numeric(df_wide['Id'], errors='coerce').astype('Int64')
    
    # 合并数据
    print("正在合并数据...")
    df_merged = pd.merge(df_wide, df_label, on='Id', how='left')
    
    # 将缺失的标签值设置为0
    df_merged['Label'] = df_merged['Label'].fillna(0)
    
    # 重新排列列的顺序，将Label列放在第二位
    columns = df_merged.columns.tolist()
    if 'Label' in columns:
        columns.remove('Label')
        # 将Label列插入到第二位（索引为1）
        columns.insert(1, 'Label')
        df_merged = df_merged[columns]
    
    # 查看合并后的数据信息
    print(f"合并后数据形状: {df_merged.shape}")
    print(f"Label列非空值数量: {df_merged['Label'].notna().sum()}")
    print(f"Label值分布:\n{df_merged['Label'].value_counts(dropna=False)}")
    
    # 保存带标签的大宽表
    output_path = './output/ml_wide_table_with_label.csv'
    df_merged.to_csv(output_path, index=False)
    print(f"带标签的大宽表已保存到: {output_path}")
    
    # 参考fake.py，将大宽表的真实数据复制到train.csv和test.csv
    # 保存到data/train.csv
    train_path = './data/train.csv'
    df_merged.to_csv(train_path, index=False)
    print(f"带标签的大宽表已保存到: {train_path}")
    
    # 复制前100条数据到test.csv，并删除Label字段
    test_df = df_merged.head(100).drop('Label', axis=1)
    test_path = './data/test.csv'
    test_df.to_csv(test_path, index=False)
    print(f"前100条数据（无标签）已保存到: {test_path}")
    
    return df_merged

if __name__ == "__main__":
    # 从lable_key.csv读取配置
    try:
        label_config = pd.read_csv('./config/lable_key.csv')
        if not label_config.empty:
            # 使用配置文件中的第一个条目作为默认参数
            excel_file_path = f'./label/{label_config.iloc[0]["file_name"]}'
            sheet_name = label_config.iloc[0]["sheet_name"]
            label_column = label_config.iloc[0]["label_key"]
        else:
            # 如果配置文件为空，使用默认参数
            excel_file_path = './label/现金管理客户数全量2025_08（销户因素调整后).xlsx'
            sheet_name = '2024Raw'
            label_column = '本地支薪'
    except FileNotFoundError:
        # 如果配置文件不存在，使用默认参数
        excel_file_path = './label/现金管理客户数全量2025_08（销户因素调整后).xlsx'
        sheet_name = '2024Raw'
        label_column = '本地支薪'
    except Exception as e:
        # 如果读取配置文件时出现其他错误，使用默认参数
        print(f"读取配置文件时出现错误: {e}")
        excel_file_path = './label/现金管理客户数全量2025_08（销户因素调整后).xlsx'
        sheet_name = '2024Raw'
        label_column = '本地支薪'
    
    # 如果提供了命令行参数，则使用参数覆盖配置文件中的值
    if len(sys.argv) > 1:
        excel_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        sheet_name = sys.argv[2]
    if len(sys.argv) > 3:
        label_column = sys.argv[3]
    
    # 执行函数
    result_df = add_label_to_wide_table(excel_file_path, sheet_name, label_column)
