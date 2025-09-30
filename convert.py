import pandas as pd
import os
import glob
from collections import defaultdict
import psutil
import gc

# ==============================
# 1. 路径设置
# ==============================
input_dir = "data"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def get_primary_key(all_data):
    """
    优先从 config/primary_key.csv 读取主键字段，若不存在则自动识别
    """
    config_dir = "config"
    primary_key_file = os.path.join(config_dir, "primary_key.csv")
    if os.path.exists(primary_key_file):
        try:
            df_pk = pd.read_csv(primary_key_file)
            # 支持以下两种格式：
            # 1. 第一列叫 primary_key，内容是主键名
            # 2. 只有一个字段（无表头），第一行内容是主键名
            if 'primary_key' in df_pk.columns:
                primary_key = str(df_pk['primary_key'].iloc[0]).strip()
            else:
                primary_key = str(df_pk.iloc[0, 0]).strip()
            print(f"从配置文件读取主键字段: {primary_key}")
            return primary_key
        except Exception as e:
            print(f"读取主键配置文件 {primary_key_file} 时出错: {e}")
            # 继续自动识别

    # 自动识别主键字段
    primary_key_candidates = ['cusno', 'ci', '客户编号', '客户号']
    for file_name, dataframes in all_data.items():
        for df in dataframes:
            for col in df.columns:
                if any(candidate in col.lower() for candidate in primary_key_candidates):
                    print(f"自动识别到主键字段: {col}")
                    return col
    print("未找到主键字段，使用索引作为主键")
    return 'index'

def process_all_excel_files():
    """
    通用机器学习宽表转化器
    1. 读取data/目录下的所有Excel文件
    2. 自动学习各文件的字段，分析各种维度
    3. 根据各种维度对数据进行透视，生成用于机器学习的宽表
    4. 自动计算一些衍生特征
    5. 将最终的宽表和字段描述分别保存为CSV文件到output/目录中
    """
    # 获取所有Excel文件
    excel_files = glob.glob(os.path.join(input_dir, "*.xlsx"))
    print(f"找到 {len(excel_files)} 个Excel文件")
    
    # 存储所有数据的字典
    all_data = defaultdict(list)
    
    # 读取所有Excel文件
    for file_path in excel_files:
        print(f"正在处理文件: {os.path.basename(file_path)}")
        try:
            # 读取Excel文件的所有sheet
            excel_data = pd.read_excel(file_path, sheet_name=None)
            
            # 遍历每个sheet
            for sheet_name, df in excel_data.items():
                # 标准化列名（转换为小写并去除空格）
                df.columns = [col.strip().lower() if isinstance(col, str) else col for col in df.columns]
                
                # 添加文件来源列
                df['source_file'] = os.path.basename(file_path)
                df['sheet_name'] = sheet_name
                
                # 存储数据
                all_data[os.path.basename(file_path)].append(df)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    return all_data

def analyze_fields_and_dimensions(all_data):
    """
    自动学习字段和分析维度
    """
    # 分析所有字段
    field_analysis = {}
    dimension_analysis = {}
    
    for file_name, dataframes in all_data.items():
        for df in dataframes:
            # 分析字段
            for col in df.columns:
                if col not in field_analysis:
                    field_analysis[col] = {
                        'data_types': set(),
                        'files': set(),
                        'sample_values': set()
                    }
                
                # 记录数据类型
                field_analysis[col]['data_types'].add(str(df[col].dtype))
                
                # 记录出现的文件
                field_analysis[col]['files'].add(file_name)
                
                # 记录样本值
                sample_vals = df[col].dropna().unique()[:5]  # 取前5个唯一值作为样本
                field_analysis[col]['sample_values'].update(sample_vals)
            
            # 分析可能的维度（数值型字段除外）
            for col in df.columns:
                if (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])) and col not in ['source_file', 'sheet_name']:
                    if col not in dimension_analysis:
                        dimension_analysis[col] = {
                            'files': set(),
                            'values': set()
                        }
                    
                    # 记录出现的文件
                    dimension_analysis[col]['files'].add(file_name)
                    
                    # 记录唯一值
                    unique_vals = df[col].dropna().unique()
                    dimension_analysis[col]['values'].update(unique_vals[:100])  # 限制存储数量
    
    return field_analysis, dimension_analysis

def create_wide_table(all_data, dimension_analysis):
    """
    根据维度创建宽表
    """
    # 主键字段
    primary_key = get_primary_key(all_data)
    print(f"使用主键字段: {primary_key}")
    
    # 创建宽表
    wide_dfs = []
    
    # 遍历每个文件和sheet
    for file_name, dataframes in all_data.items():
        for df in dataframes:
            # 如果没有主键字段，则添加索引
            if primary_key not in df.columns:
                df[primary_key] = df.index.astype(str)
            
            # 获取数值型字段用于聚合
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != primary_key]
            
            # 获取维度字段
            dimension_cols = [col for col in df.columns 
                            if (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])) 
                            and col not in ['source_file', 'sheet_name', primary_key]]
            
            # 对每个维度字段进行透视
            for dim_col in dimension_cols:
                if dim_col in dimension_analysis and len(dimension_analysis[dim_col]['values']) <= 50:  # 降低维度值数量限制
                    print(f"正在处理维度: {dim_col} (唯一值数量: {len(dimension_analysis[dim_col]['values'])})")
                    
                    for numeric_col in numeric_cols:
                        try:
                            max_rows = 50000
                            if len(df) > max_rows:
                                df_subset = df.iloc[:max_rows]
                            else:
                                df_subset = df
                            
                            # 创建透视表
                            pivot = df_subset.pivot_table(
                                index=primary_key,
                                columns=dim_col,
                                values=numeric_col,
                                aggfunc='sum',
                                fill_value=0
                            )
                            
                            # 重命名列
                            pivot.columns = [f"{numeric_col}_{dim_col}_{col}" for col in pivot.columns]
                            
                            # 重置索引
                            pivot = pivot.reset_index()
                            
                            # 添加标识列
                            pivot['source_file'] = file_name
                            pivot['dimension'] = dim_col
                            pivot['value_field'] = numeric_col
                            
                            wide_dfs.append(pivot)
                            
                            # 监控内存使用情况
                            process = psutil.Process(os.getpid())
                            memory_info = process.memory_info()
                            if memory_info.rss / psutil.virtual_memory().total > 0.7:
                                gc.collect()
                                print("执行垃圾回收")
                                
                            del pivot
                            del df_subset
                        except Exception as e:
                            print(f"创建透视表时出错 ({dim_col}, {numeric_col}): {e}")
                            gc.collect()
                else:
                    if dim_col in dimension_analysis:
                        print(f"维度 {dim_col} 的唯一值数量过多 ({len(dimension_analysis[dim_col]['values'])})，跳过处理")
                    else:
                        print(f"维度 {dim_col} 未在维度分析中找到")
    
    # 合并所有宽表
    if wide_dfs:
        print(f"共有 {len(wide_dfs)} 个数据框需要合并")
        
        batch_size = 10
        merged_dfs = []
        
        for i in range(0, len(wide_dfs), batch_size):
            batch = wide_dfs[i:i+batch_size]
            batch_merged = pd.concat(batch, axis=0, sort=False)
            merged_dfs.append(batch_merged)
            print(f"合并批次 {i//batch_size + 1}/{(len(wide_dfs)-1)//batch_size + 1}")
            gc.collect()
        
        merged_df = pd.concat(merged_dfs, axis=0, sort=False)
        del merged_dfs
        gc.collect()
        
        numeric_columns = [col for col in merged_df.columns if pd.api.types.is_numeric_dtype(merged_df[col])]
        other_columns = [col for col in merged_df.columns if col not in numeric_columns]
        
        max_agg_cols = 1000
        if len(numeric_columns) > max_agg_cols:
            print(f"数值列数量过多 ({len(numeric_columns)})，仅处理前 {max_agg_cols} 列")
            numeric_columns = numeric_columns[:max_agg_cols]
        
        agg_dict = {}
        for col in numeric_columns:
            if col != primary_key:
                agg_dict[col] = 'sum'
        
        for col in other_columns:
            if col != primary_key and len(agg_dict) < max_agg_cols:
                agg_dict[col] = 'first'
        
        print("正在进行分组聚合...")
        final_df = merged_df.groupby(primary_key).agg(agg_dict).reset_index()
        del merged_df
        gc.collect()
        
        final_numeric_columns = [col for col in final_df.columns if pd.api.types.is_numeric_dtype(final_df[col])]
        final_df[final_numeric_columns] = final_df[final_numeric_columns].fillna(0)
        
        final_other_columns = [col for col in final_df.columns if col not in final_numeric_columns]
        for col in final_other_columns:
            if col != primary_key:
                final_df[col] = final_df[col].fillna('Unknown')
        
        if primary_key in final_df.columns:
            final_df.rename(columns={primary_key: 'Id'}, inplace=True)
        
        if 'source_file' in final_df.columns:
            final_df.drop(columns=['source_file'], inplace=True)
        
        return final_df
    else:
        print("未能创建任何透视表")
        return pd.DataFrame()

def calculate_derived_features(wide_df):
    """
    计算衍生特征
    """
    if wide_df.empty:
        return wide_df
    
    # 获取所有数值型字段（排除标识列）
    numeric_cols = [col for col in wide_df.columns if pd.api.types.is_numeric_dtype(wide_df[col])]
    exclude_cols = ['ci', 'cusno', 'index']  # 可能的主键列
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    max_feature_cols = 500
    if len(numeric_cols) > max_feature_cols:
        print(f"数值列数量过多 ({len(numeric_cols)})，仅使用前 {max_feature_cols} 列计算衍生特征")
        numeric_cols = numeric_cols[:max_feature_cols]
    
    new_features = {}
    
    if len(numeric_cols) > 0:
        print("正在计算总和特征...")
        new_features['total_sum'] = wide_df[numeric_cols].sum(axis=1)
    
    if len(numeric_cols) > 1:
        print("正在计算统计特征...")
        new_features['total_mean'] = wide_df[numeric_cols].mean(axis=1)
        new_features['total_std'] = wide_df[numeric_cols].std(axis=1)
        new_features['total_max'] = wide_df[numeric_cols].max(axis=1)
        new_features['total_min'] = wide_df[numeric_cols].min(axis=1)
    
    if new_features:
        new_features_df = pd.DataFrame(new_features, index=wide_df.index)
        wide_df = pd.concat([wide_df, new_features_df], axis=1)
    
    print(f"计算了衍生特征，当前宽表形状: {wide_df.shape}")
    return wide_df

def generate_feature_dictionary(wide_df):
    """
    生成字段描述字典
    """
    feature_dict = []
    
    for col in wide_df.columns:
        # 用 pandas API 判断字段类型
        if pd.api.types.is_numeric_dtype(wide_df[col]):
            feature_type = 'continuous'
        elif pd.api.types.is_string_dtype(wide_df[col]) or pd.api.types.is_object_dtype(wide_df[col]):
            if wide_df[col].nunique() <= 10:
                feature_type = 'category'
            else:
                feature_type = 'text'
        else:
            feature_type = 'other'
        
        feature_dict.append({
            'feature_name': col,
            'feature_type': feature_type
        })
    
    return pd.DataFrame(feature_dict)

def main():
    # 处理所有Excel文件
    all_data = process_all_excel_files()
    
    # 分析字段和维度
    field_analysis, dimension_analysis = analyze_fields_and_dimensions(all_data)
    print(f"分析了 {len(field_analysis)} 个字段, {len(dimension_analysis)} 个维度")
    
    # 创建宽表
    wide_df = create_wide_table(all_data, dimension_analysis)
    print(f"宽表形状: {wide_df.shape}")
    
    # 计算衍生特征
    wide_df = calculate_derived_features(wide_df)
    
    # 生成字段字典
    feature_dict_df = generate_feature_dictionary(wide_df)
    
    # 保存结果
    output_csv = os.path.join(output_dir, "ml_wide_table.csv")
    output_dict = os.path.join(output_dir, "feature_dictionary.csv")
    
    wide_df.to_csv(output_csv, index=False, encoding='utf-8')
    feature_dict_df.to_csv(output_dict, index=False, encoding='utf-8')
    
    # 复制 feature_dictionary.csv 到 config/features.csv，并删除类型为 text 的记录行
    config_dir = "config"
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "features.csv")
    
    feature_dict_df_filtered = feature_dict_df[feature_dict_df['feature_type'] != 'text']
    feature_dict_df_filtered.to_csv(config_file, index=False, encoding='utf-8')
    
    print(f"✅ 宽表已保存到: {output_csv} (UTF-8)")
    print(f"✅ 字段描述已保存到: {output_dict} (UTF-8)")
    print(f"✅ 过滤后的字段描述已保存到: {config_file} (UTF-8)")
    print(f"\n📊 最终数据形状: {wide_df.shape[0]} 行, {wide_df.shape[1]} 列")

if __name__ == "__main__":
    main()
