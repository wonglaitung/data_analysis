import pandas as pd
import os
import glob
import re
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
    读取所有Excel文件，为每个文件字段加唯一前缀（主键和元字段除外）
    返回: all_data (按文件组织), all_primary_keys (小写集合)
    """
    excel_files = glob.glob(os.path.join(input_dir, "*.xlsx"))
    print(f"找到 {len(excel_files)} 个Excel文件")
    all_data = defaultdict(list)
    all_primary_keys_set = set(['__primary_key__', 'id', 'cusno', 'ci', 'index'])  # 小写

    primary_key_mapping = get_primary_key_mapping()

    for file_path in excel_files:
        file_name = os.path.basename(file_path)
        print(f"正在处理文件: {file_name}")
        try:
            excel_data = pd.read_excel(file_path, sheet_name=None)
            for sheet_name, df in excel_data.items():
                # 列名转小写并去空格
                df.columns = [col.strip().lower() if isinstance(col, str) else col for col in df.columns]
                df['source_file'] = file_name
                df['sheet_name'] = sheet_name

                # 生成安全前缀（移除 .xlsx 和非法字符）
                safe_prefix = re.sub(r'[^\w]', '_', file_name.replace('.xlsx', ''))
                # 为非元字段加前缀
                new_columns = []
                for col in df.columns:
                    if col in ['__primary_key__', 'source_file', 'sheet_name']:
                        new_columns.append(col)
                    else:
                        new_columns.append(f"{safe_prefix}_{col}")
                df.columns = new_columns

                # 确定主键
                pk = None
                if (file_name, sheet_name) in primary_key_mapping:
                    pk = primary_key_mapping[(file_name, sheet_name)].strip()
                elif (file_name, '') in primary_key_mapping:
                    pk = primary_key_mapping[(file_name, '')].strip()
                elif auto_detect_key(df):
                    pk = auto_detect_key(df)
                else:
                    pk = 'index'
                    df[pk] = df.index.astype(str)

                pk_lower = pk.lower()
                if pk_lower in df.columns:
                    df['__primary_key__'] = df[pk_lower]
                else:
                    df['__primary_key__'] = df[pk]

                print(f"  使用主键: {pk}")
                all_primary_keys_set.add(pk_lower)
                all_data[file_name].append(df)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    return all_data, [k.lower() for k in all_primary_keys_set]

def analyze_fields_and_dimensions(all_data):
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
    total = value_counts.sum()
    if total == 0:
        return []
    cumulative = value_counts.cumsum() / total
    topk_idx = cumulative[cumulative <= coverage_threshold].index.tolist()
    if len(topk_idx) < len(value_counts):
        topk_idx.append(cumulative.index[len(topk_idx)])
    if len(topk_idx) > max_top_k:
        topk_idx = topk_idx[:max_top_k]
    return topk_idx

def create_wide_table_per_file(all_data, dimension_analysis, all_primary_keys, coverage_threshold=0.95, max_top_k=50):
    file_wide_tables = {}

    for file_name, dataframes in all_data.items():
        print(f"\n=== 处理文件宽表: {file_name} ===")
        file_wide_dfs = []

        for df in dataframes:
            primary_key = '__primary_key__'
            if primary_key not in df.columns:
                df[primary_key] = df.index.astype(str)

            numeric_cols = [
                col for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col])
                and col != primary_key
                and col.lower() not in all_primary_keys
            ]
            dimension_cols = [
                col for col in df.columns
                if ((pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]))
                    and col not in ['source_file', 'sheet_name']
                    and col.lower() not in all_primary_keys)
            ]

            for dim_col in dimension_cols:
                if dim_col in dimension_analysis:
                    value_counts = df[dim_col].value_counts()
                    topk_values = get_topk_by_coverage(value_counts, coverage_threshold=coverage_threshold, max_top_k=max_top_k)
                    df[dim_col] = df[dim_col].where(df[dim_col].isin(topk_values), other='other')

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

                            file_wide_dfs.append(pivot)

                            process = psutil.Process(os.getpid())
                            memory_info = process.memory_info()
                            if memory_info.rss / psutil.virtual_memory().total > 0.7:
                                gc.collect()
                                print("  执行垃圾回收")

                            del pivot, df_subset
                        except Exception as e:
                            print(f"  警告：透视失败 ({dim_col}, {numeric_col}): {e}")
                else:
                    print(f"  跳过未分析维度: {dim_col}")

        if not file_wide_dfs:
            print(f"  ⚠️ 文件 {file_name} 未生成任何透视表")
            continue

        print(f"  合并 {len(file_wide_dfs)} 个透视表...")
        merged_df = pd.concat(file_wide_dfs, axis=0, sort=False)
        del file_wide_dfs
        gc.collect()

        numeric_columns = [col for col in merged_df.columns if pd.api.types.is_numeric_dtype(merged_df[col])]
        other_columns = [col for col in merged_df.columns if col not in numeric_columns]

        max_agg_cols = 1000
        if len(numeric_columns) > max_agg_cols:
            numeric_columns = numeric_columns[:max_agg_cols]

        agg_dict = {}
        for col in numeric_columns:
            if col != '__primary_key__':
                agg_dict[col] = 'sum'
        for col in other_columns:
            if col != '__primary_key__' and len(agg_dict) < max_agg_cols:
                agg_dict[col] = 'first'

        final_df = merged_df.groupby('__primary_key__').agg(agg_dict).reset_index()
        del merged_df
        gc.collect()

        numeric_cols_final = [col for col in final_df.columns if pd.api.types.is_numeric_dtype(final_df[col])]
        other_cols_final = [col for col in final_df.columns if col not in numeric_cols_final]

        final_df[numeric_cols_final] = final_df[numeric_cols_final].fillna(0)
        for col in other_cols_final:
            if col != '__primary_key__':
                final_df[col] = final_df[col].fillna('Unknown')

        if '__primary_key__' in final_df.columns:
            final_df.rename(columns={'__primary_key__': 'Id'}, inplace=True)
        if 'source_file' in final_df.columns:
            final_df.drop(columns=['source_file'], inplace=True)

        file_wide_tables[file_name] = final_df
        print(f"  ✅ 完成文件 {file_name}，宽表形状: {final_df.shape}")

    return file_wide_tables

def calculate_derived_features(wide_df):
    if wide_df.empty:
        return wide_df
    numeric_cols = [col for col in wide_df.columns if pd.api.types.is_numeric_dtype(wide_df[col])]
    exclude_cols = ['ci', 'cusno', 'index', 'Id']
    numeric_cols = [col for col in numeric_cols if col.lower() not in [x.lower() for x in exclude_cols]]
    max_feature_cols = 500
    if len(numeric_cols) > max_feature_cols:
        numeric_cols = numeric_cols[:max_feature_cols]
    new_features = {}
    if len(numeric_cols) > 0:
        new_features['total_sum'] = wide_df[numeric_cols].sum(axis=1)
    if len(numeric_cols) > 1:
        new_features['total_mean'] = wide_df[numeric_cols].mean(axis=1)
        new_features['total_std'] = wide_df[numeric_cols].std(axis=1)
        new_features['total_max'] = wide_df[numeric_cols].max(axis=1)
        new_features['total_min'] = wide_df[numeric_cols].min(axis=1)
    if new_features:
        new_features_df = pd.DataFrame(new_features, index=wide_df.index)
        wide_df = pd.concat([wide_df, new_features_df], axis=1)
    print(f"  衍生特征计算完成，当前形状: {wide_df.shape}")
    return wide_df

def generate_feature_dictionary(wide_df):
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
    all_data, all_primary_keys = process_all_excel_files()
    field_analysis, dimension_analysis = analyze_fields_and_dimensions(all_data)
    print(f"\n分析了 {len(field_analysis)} 个字段, {len(dimension_analysis)} 个维度")

    file_wide_tables = create_wide_table_per_file(
        all_data,
        dimension_analysis,
        all_primary_keys,
        coverage_threshold=coverage_threshold,
        max_top_k=max_top_k
    )

    if not file_wide_tables:
        print("❌ 未生成任何宽表")
        return

    # === 保存每个文件的独立宽表（溯源用）===
    for file_name, wide_df in file_wide_tables.items():
        safe_name = re.sub(r'[^\w]', '_', file_name.replace('.xlsx', ''))
        output_csv = os.path.join(output_dir, f"wide_table_{safe_name}.csv")
        wide_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"✅ 保存文件宽表: {output_csv}")

        feature_dict_df = generate_feature_dictionary(wide_df)
        dict_csv = os.path.join(output_dir, f"feature_dict_{safe_name}.csv")
        feature_dict_df.to_csv(dict_csv, index=False, encoding='utf-8')

    # === 合并所有文件宽表（建模用）===
    print("\n=== 合并所有文件宽表（用于建模）===")
    all_wide_dfs = list(file_wide_tables.values())
    if len(all_wide_dfs) == 1:
        global_wide = all_wide_dfs[0].copy()
    else:
        global_wide = all_wide_dfs[0].copy()
        for df in all_wide_dfs[1:]:
            global_wide = pd.merge(global_wide, df, on='Id', how='outer')

    global_wide = calculate_derived_features(global_wide)

    global_output = os.path.join(output_dir, "ml_wide_table_global.csv")
    global_wide.to_csv(global_output, index=False, encoding='utf-8')

    global_dict = generate_feature_dictionary(global_wide)
    global_dict.to_csv(os.path.join(output_dir, "feature_dictionary_global.csv"), index=False, encoding='utf-8')

    config_dir = "config"
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "features.csv")
    global_dict_filtered = global_dict[global_dict['feature_type'] != 'text']
    global_dict_filtered.to_csv(config_file, index=False, encoding='utf-8')

    print(f"\n✅ 全局宽表已保存: {global_output}")
    print(f"✅ 全局字段字典: {os.path.join(output_dir, 'feature_dictionary_global.csv')}")
    print(f"✅ 建模用特征列表: {config_file}")
    print(f"\n📊 全局宽表最终形状: {global_wide.shape[0]} 行, {global_wide.shape[1]} 列")

if __name__ == "__main__":
    main(coverage_threshold=0.95, max_top_k=50)
