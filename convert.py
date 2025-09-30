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

def get_primary_key_mapping():
    """
    读取 config/primary_key.csv，返回 {(file_name, sheet_name): primary_key} 映射
    支持 sheet_name 为空，表示该文件所有sheet都用此主键
    """
    config_dir = "config"
    primary_key_file = os.path.join(config_dir, "primary_key.csv")
    mapping = {}
    if os.path.exists(primary_key_file):
        try:
            df_pk = pd.read_csv(primary_key_file, dtype=str)
            for _, row in df_pk.iterrows():
                file = str(row.get('file_name', '')).strip()
                sheet = str(row.get('sheet_name', '')).strip() if 'sheet_name' in row else ''
                pk = str(row.get('primary_key', '')).strip()
                if file and pk:
                    mapping[(file, sheet)] = pk
        except Exception as e:
            print(f"读取主键配置文件 {primary_key_file} 时出错: {e}")
    return mapping

def auto_detect_key(df):
    """
    自动检测主键列名
    """
    primary_key_candidates = ['cusno', 'ci', '客户编号', '客户号']
    for col in df.columns:
        if any(candidate in col.lower() for candidate in primary_key_candidates):
            return col
    return None

def process_all_excel_files():
    """
    通用机器学习宽表转化器
    1. 读取data/目录下的所有Excel文件
    2. 自动学习各文件的字段，分析各种维度
    3. 根据各种维度对数据进行透视，生成用于机器学习的宽表
    4. 自动计算一些衍生特征
    5. 将最终的宽表和字段描述分别保存为CSV文件到output/目录中
    """
    excel_files = glob.glob(os.path.join(input_dir, "*.xlsx"))
    print(f"找到 {len(excel_files)} 个Excel文件")
    all_data = defaultdict(list)
    primary_key_mapping = get_primary_key_mapping()

    for file_path in excel_files:
        file_name = os.path.basename(file_path)
        print(f"正在处理文件: {file_name}")
        try:
            excel_data = pd.read_excel(file_path, sheet_name=None)
            for sheet_name, df in excel_data.items():
                df.columns = [col.strip().lower() if isinstance(col, str) else col for col in df.columns]
                df['source_file'] = file_name
                df['sheet_name'] = sheet_name

                # 主键设定逻辑
                pk = None
                # 优先查 file+sheet 配置
                if (file_name, sheet_name) in primary_key_mapping:
                    pk = primary_key_mapping[(file_name, sheet_name)]
                # 再查 file+空sheet 配置
                elif (file_name, '') in primary_key_mapping:
                    pk = primary_key_mapping[(file_name, '')]
                # 自动检测
                elif auto_detect_key(df):
                    pk = auto_detect_key(df)
                else:
                    pk = 'index'
                    df[pk] = df.index.astype(str)
                df['__primary_key__'] = df[pk] if pk in df.columns else df[pk]

                all_data[file_name].append(df)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    return all_data

def analyze_fields_and_dimensions(all_data):
    """
    自动学习字段和分析维度
    """
    field_analysis = {}
    dimension_analysis = {}
    for file_name, dataframes in all_data.items():
        for df in dataframes:
            for col in df.columns:
                if col not in field_analysis:
                    field_analysis[col] = {
                        'data_types': set(),
                        'files': set(),
                        'sample_values': set()
                    }
                field_analysis[col]['data_types'].add(str(df[col].dtype))
                field_analysis[col]['files'].add(file_name)
                sample_vals = df[col].dropna().unique()[:5]
                field_analysis[col]['sample_values'].update(sample_vals)
            for col in df.columns:
                if (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])) and col not in ['source_file', 'sheet_name']:
                    if col not in dimension_analysis:
                        dimension_analysis[col] = {
                            'files': set(),
                            'values': set()
                        }
                    dimension_analysis[col]['files'].add(file_name)
                    unique_vals = df[col].dropna().unique()
                    dimension_analysis[col]['values'].update(unique_vals[:100])
    return field_analysis, dimension_analysis

def get_topk_by_coverage(value_counts, coverage_threshold=0.95, max_top_k=50):
    """
    根据累计覆盖率和max_top_k，返回top_k类别列表
    """
    total = value_counts.sum()
    cumulative = value_counts.cumsum() / total
    topk_idx = cumulative[cumulative <= coverage_threshold].index.tolist()
    if len(topk_idx) < len(value_counts):
        # 包含第一个超过阈值的类别
        topk_idx.append(cumulative.index[len(topk_idx)])
    if len(topk_idx) > max_top_k:
        topk_idx = topk_idx[:max_top_k]
    return topk_idx

def create_wide_table(all_data, dimension_analysis, coverage_threshold=0.95, max_top_k=50):
    """
    根据维度创建宽表，采用覆盖率阈值+最大Top-K策略防止维度膨胀
    """
    wide_dfs = []

    for file_name, dataframes in all_data.items():
        for df in dataframes:
            primary_key = '__primary_key__'
            if primary_key not in df.columns:
                df[primary_key] = df.index.astype(str)

            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != primary_key]
            dimension_cols = [col for col in df.columns
                              if (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]))
                              and col not in ['source_file', 'sheet_name', primary_key]]

            for dim_col in dimension_cols:
                if dim_col in dimension_analysis:
                    value_counts = df[dim_col].value_counts()
                    topk_values = get_topk_by_coverage(value_counts, coverage_threshold=coverage_threshold, max_top_k=max_top_k)
                    df[dim_col] = df[dim_col].where(df[dim_col].isin(topk_values), other='other')
                    unique_vals = df[dim_col].unique()
                    cumulative = value_counts.cumsum() / value_counts.sum()
                    coverage_info = cumulative[topk_values[-1]] if topk_values else 0
                    print(f"维度: {dim_col}，Top-{len(topk_values)}值: {topk_values}, 其它归为 'other', 总列数: {len(unique_vals)}, 覆盖率: {coverage_info:.2%}")

                    for numeric_col in numeric_cols:
                        try:
                            max_rows = 50000
                            df_subset = df.iloc[:max_rows] if len(df) > max_rows else df

                            pivot = df_subset.pivot_table(
                                index=primary_key,
                                columns=dim_col,
                                values=numeric_col,
                                aggfunc='sum',
                                fill_value=0
                            )

                            pivot.columns = [f"{numeric_col}_{dim_col}_{col}" for col in pivot.columns]
                            pivot = pivot.reset_index()
                            pivot['source_file'] = file_name
                            pivot['dimension'] = dim_col
                            pivot['value_field'] = numeric_col

                            wide_dfs.append(pivot)

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
                    print(f"维度 {dim_col} 未在维度分析中找到")

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
            if col != '__primary_key__':
                agg_dict[col] = 'sum'
        for col in other_columns:
            if col != '__primary_key__' and len(agg_dict) < max_agg_cols:
                agg_dict[col] = 'first'
        
        print("正在进行分组聚合...")
        final_df = merged_df.groupby('__primary_key__').agg(agg_dict).reset_index()
        del merged_df
        gc.collect()
        
        final_numeric_columns = [col for col in final_df.columns if pd.api.types.is_numeric_dtype(final_df[col])]
        final_df[final_numeric_columns] = final_df[final_numeric_columns].fillna(0)
        final_other_columns = [col for col in final_df.columns if col not in final_numeric_columns]
        for col in final_other_columns:
            if col != '__primary_key__':
                final_df[col] = final_df[col].fillna('Unknown')
        if '__primary_key__' in final_df.columns:
            final_df.rename(columns={'__primary_key__': 'Id'}, inplace=True)
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
    numeric_cols = [col for col in wide_df.columns if pd.api.types.is_numeric_dtype(wide_df[col])]
    exclude_cols = ['ci', 'cusno', 'index', 'Id']
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

def main(coverage_threshold=0.95, max_top_k=50):
    all_data = process_all_excel_files()
    field_analysis, dimension_analysis = analyze_fields_and_dimensions(all_data)
    print(f"分析了 {len(field_analysis)} 个字段, {len(dimension_analysis)} 个维度")
    wide_df = create_wide_table(all_data, dimension_analysis, coverage_threshold=coverage_threshold, max_top_k=max_top_k)
    print(f"宽表形状: {wide_df.shape}")
    wide_df = calculate_derived_features(wide_df)
    feature_dict_df = generate_feature_dictionary(wide_df)
    output_csv = os.path.join(output_dir, "ml_wide_table.csv")
    output_dict = os.path.join(output_dir, "feature_dictionary.csv")
    wide_df.to_csv(output_csv, index=False, encoding='utf-8')
    feature_dict_df.to_csv(output_dict, index=False, encoding='utf-8')
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
    main(coverage_threshold=0.95, max_top_k=50)
