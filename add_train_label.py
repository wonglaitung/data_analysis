import pandas as pd
import sys
import os

def balance_samples(df, target_column='Label', positive_negative_ratio=1.0):
    """
    平衡正负样本，保留所有正样本，按比例减少负样本
    
    参数:
    df: 包含标签的数据框
    target_column: 标签列名，默认为'Label'
    positive_negative_ratio: 正负样本目标比例，默认为1.0（即1:1）
    
    返回:
    平衡后的数据框
    """
    # 分离正负样本
    positive_samples = df[df[target_column] == 1]
    negative_samples = df[df[target_column] == 0]
    
    print(f"原始样本分布: 正样本 {len(positive_samples)} 个, 负样本 {len(negative_samples)} 个")
    
    # 计算需要的负样本数量
    target_negative_count = int(len(positive_samples) * positive_negative_ratio)
    
    # 如果负样本数量已经少于目标数量，则不进行下采样
    if len(negative_samples) <= target_negative_count:
        print("负样本数量已少于或等于目标数量，无需下采样")
        return df
    
    # 随机采样负样本
    sampled_negative_samples = negative_samples.sample(n=target_negative_count, random_state=42)
    
    # 合并正样本和采样后的负样本
    balanced_df = pd.concat([positive_samples, sampled_negative_samples], ignore_index=True)
    
    # 打乱数据顺序
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"平衡后样本分布: 正样本 {len(positive_samples)} 个, 负样本 {len(sampled_negative_samples)} 个")
    
    return balanced_df

def add_label_to_wide_table(excel_file_path, sheet_name='2024Raw', label_column='本地支薪', balance_ratio=None):
    """
    将标签添加到大宽表
    
    参数:
    excel_file_path: Excel文件路径
    sheet_name: 工作表名称，默认为'2024Raw'
    label_column: 用作标签的列名，默认为'本地支薪'
    balance_ratio: 正负样本平衡比例（正样本:负样本），默认为None（不进行平衡）
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
    matched_ids = len(pd.merge(df_wide[['Id']], df_label[['Id']], on='Id', how='inner'))
    total_wide_ids = len(df_wide)
    match_rate = matched_ids / total_wide_ids if total_wide_ids > 0 else 0
    print(f"基于主键(Id)匹配率: {match_rate:.2%} ({matched_ids}/{total_wide_ids})")
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
    
    # 如果指定了平衡比例，则进行样本平衡处理
    if balance_ratio is not None:
        print(f"正在进行样本平衡处理，目标正负样本比例: {balance_ratio}:1")
        df_merged = balance_samples(df_merged, 'Label', balance_ratio)
        print(f"平衡后数据形状: {df_merged.shape}")
        print(f"平衡后Label值分布:\n{df_merged['Label'].value_counts(dropna=False)}")
    
    # 保存带标签的大宽表
    output_path = './output/ml_wide_table_with_label.csv'
    df_merged.to_csv(output_path, index=False)
    print(f"带标签的大宽表已保存到: {output_path}")
    
    # 参考fake.py，将大宽表的真实数据复制到train.csv和test.csv
    # 保存到data_train/train.csv
    train_path = './data_train/train.csv'
    df_merged.to_csv(train_path, index=False)
    print(f"带标签的大宽表已保存到: {train_path}")
    
    # 复制前100条数据到test.csv，并删除Label字段
    test_df = df_merged.head(100).drop('Label', axis=1)
    test_path = './data_train/test.csv'
    test_df.to_csv(test_path, index=False)
    print(f"前100条数据（无标签）已保存到: {test_path}")
    
    return df_merged

def print_usage():
    """
    打印使用说明
    """
    print("使用方法:")
    print("  python add_train_label.py [Excel文件路径] [工作表名称] [标签列名] [--balance-ratio 比例]")
    print("")
    print("参数说明:")
    print("  Excel文件路径    Excel文件的路径，默认从config/label_key.csv读取")
    print("  工作表名称       Excel文件中的工作表名称，默认为'2024Raw'")
    print("  标签列名         用作标签的列名，默认为'本地支薪'")
    print("  --balance-ratio  正负样本平衡比例（正样本:负样本），默认为None（不进行平衡）")
    print("")
    print("示例:")
    print("  python add_train_label.py")
    print("  python add_train_label.py --balance-ratio 1.0")
    print("  python add_train_label.py ./label_train/标签文件.xlsx 2024Raw 本地支薪 --balance-ratio 0.5")

if __name__ == "__main__":
    # 检查是否有--help参数
    if '--help' in sys.argv or '-h' in sys.argv:
        print_usage()
        sys.exit(0)
    
    # 从label_key.csv读取配置
    label_config_file = './config/label_key.csv'
    try:
        label_config = pd.read_csv(label_config_file)
        if not label_config.empty:
            # 使用配置文件中的第一个条目作为默认参数
            excel_file_path = f'./label_train/{label_config.iloc[0]["file_name"]}'
            sheet_name = label_config.iloc[0]["sheet_name"]
            label_column = label_config.iloc[0]["label_key"]
        else:
            # 如果配置文件为空，使用默认参数
            excel_file_path = './label_train/现金管理客户数全量2025_08（销户因素调整后).xlsx'
            sheet_name = '2024Raw'
            label_column = '本地支薪'
    except FileNotFoundError:
        # 如果配置文件不存在，使用默认参数
        excel_file_path = './label_train/现金管理客户数全量2025_08（销户因素调整后).xlsx'
        sheet_name = '2024Raw'
        label_column = '本地支薪'
    except Exception as e:
        # 如果读取配置文件时出现其他错误，使用默认参数
        print(f"读取配置文件时出现错误: {e}")
        excel_file_path = './label_train/现金管理客户数全量2025_08（销户因素调整后).xlsx'
        sheet_name = '2024Raw'
        label_column = '本地支薪'
    
    # 解析命令行参数
    balance_ratio = None
    if '--balance-ratio' in sys.argv:
        try:
            balance_ratio_index = sys.argv.index('--balance-ratio')
            balance_ratio = float(sys.argv[balance_ratio_index + 1])
            # 从sys.argv中移除--balance-ratio及其参数
            sys.argv.pop(balance_ratio_index + 1)
            sys.argv.pop(balance_ratio_index)
        except (ValueError, IndexError):
            print("无效的平衡比例参数，将使用默认值（不进行平衡）")
    
    # 如果提供了命令行参数，则使用参数覆盖配置文件中的值
    if len(sys.argv) > 1:
        excel_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        sheet_name = sys.argv[2]
    if len(sys.argv) > 3:
        label_column = sys.argv[3]
    
    # 执行函数
    result_df = add_label_to_wide_table(excel_file_path, sheet_name, label_column, balance_ratio)