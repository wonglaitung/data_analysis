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
    primary_key_candidates = ['cusno', 'ci', '客户编号', '客户号']
    for col in df.columns:
        if any(candidate in col.lower() for candidate in primary_key_candidates):
            return col
    return None

def process_all_excel_files():
    excel_files = glob.glob(os.path.join(input_dir, "*.xlsx"))
    print(f"找到 {len(excel_files)} 个Excel文件")
    all_data = defaultdict(list)
    all_primary_keys_set = set(['__primary_key__', 'id', 'cusno', 'ci', 'index'])

    primary_key_mapping = get_primary_key_mapping()

    for file_path in excel_files:
        file_name = os.path.basename(file_path)
        print(f"正在处理文件: {file_name}")
        try:
            excel_data = pd.read_excel(file_path, sheet_name=None)
            for sheet_name, df in excel_data.items():
                # 列名转小写并去空格
                df.columns = [col.strip().lower() if isinstance(col, str) else col for col in df.columns]

                # === 关键：不再添加 source_file / sheet_name ===
                safe_prefix = re.sub(r'[^\w]', '_', file_name.replace('.xlsx', ''))
                new_columns = []
                for col in df.columns:
                    if col == '__primary_key__':
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
                # === 不再排除 source_file/sheet_name（因为它们不存在）===
                if (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])) and col != '__primary_key__':
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
        category_features = []  # 用于存储类别型特征

        safe_prefix = re.sub(r'[^\w]', '_', file_name.replace('.xlsx', ''))

        def strip_file_prefix(full_col):
            expected_start = safe_prefix + '_'
            if full_col.startswith(expected_start):
                return full_col[len(expected_start):]
            return full_col

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
                    and col != '__primary_key__'
                    and col.lower() not in all_primary_keys)
            ]

            # 保存原始的类别型特征列
            for dim_col in dimension_cols:
                if dim_col in dimension_analysis:
                    value_counts = df[dim_col].value_counts()
                    topk_values = get_topk_by_coverage(value_counts, coverage_threshold=coverage_threshold, max_top_k=max_top_k)
                    df[dim_col] = df[dim_col].where(df[dim_col].isin(topk_values), other='other')
                    
                    # 保存类别型特征列，用于后续合并
                    category_features.append(dim_col)

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

                            orig_numeric = strip_file_prefix(numeric_col)
                            orig_dim = strip_file_prefix(dim_col)
                            pivot.columns = [
                                f"{safe_prefix}_{orig_numeric}_{orig_dim}_{col}"
                                for col in pivot.columns
                            ]
                            pivot = pivot.reset_index()

                            # === 关键：不再添加 source_file / dimension / value_field ===
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

        if not file_wide_dfs and not category_features:
            print(f"  ⚠️ 文件 {file_name} 未生成任何透视表")
            continue

        final_df = None
        
        # 如果有透视表数据，则处理透视表
        if file_wide_dfs:
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
        else:
            # 如果没有透视表数据，创建一个只包含主键的DataFrame
            all_dfs = dataframes
            if all_dfs:
                first_df = all_dfs[0]
                final_df = pd.DataFrame({'__primary_key__': first_df['__primary_key__'].unique()})
        
        # 添加原始的类别型特征列（转换为数值型）
        if category_features and dataframes:
            # 获取所有数据帧中的类别特征并合并
            all_category_data = []
            for df in dataframes:
                cols_to_include = ['__primary_key__'] + [col for col in category_features if col in df.columns]
                if cols_to_include:
                    # 对类别特征进行去重和聚合
                    category_df = df[cols_to_include].copy()
                    all_category_data.append(category_df)
            
            if all_category_data:
                # 合并所有类别特征数据
                combined_category_df = pd.concat(all_category_data, axis=0, sort=False)
                
                # 对类别特征进行聚合（取第一个值）
                category_agg_dict = {}
                for col in combined_category_df.columns:
                    if col != '__primary_key__':
                        category_agg_dict[col] = 'first'
                
                if category_agg_dict:
                    unique_category_df = combined_category_df.groupby('__primary_key__').agg(category_agg_dict).reset_index()
                    
                    # 对类别特征进行One-Hot编码
                    encoded_dfs = []
                    for col in category_agg_dict.keys():
                        if col in unique_category_df.columns:
                            # 填充缺失值
                            unique_category_df[col] = unique_category_df[col].fillna('Unknown')
                            # One-Hot编码
                            onehot_df = pd.get_dummies(unique_category_df[col], prefix=col)
                            # 添加主键列
                            onehot_df['__primary_key__'] = unique_category_df['__primary_key__'].values
                            encoded_dfs.append(onehot_df)
                    
                    # 合并所有One-Hot编码的特征
                    if encoded_dfs:
                        combined_encoded_df = encoded_dfs[0]
                        for encoded_df in encoded_dfs[1:]:
                            combined_encoded_df = pd.merge(combined_encoded_df, encoded_df, on='__primary_key__', how='outer')
                        
                        # 将编码后的特征合并到最终DataFrame中
                        if final_df is not None:
                            final_df = pd.merge(final_df, combined_encoded_df, on='__primary_key__', how='left')
                        else:
                            final_df = combined_encoded_df

        if final_df is None:
            print(f"  ⚠️ 文件 {file_name} 无法生成宽表")
            continue

        # 填充类别特征的缺失值
        category_cols = [col for col in final_df.columns 
                        if (pd.api.types.is_string_dtype(final_df[col]) or pd.api.types.is_object_dtype(final_df[col]))
                        and col != '__primary_key__']
        for col in category_cols:
            final_df[col] = final_df[col].fillna('Unknown')

        if '__primary_key__' in final_df.columns:
            final_df.rename(columns={'__primary_key__': 'Id'}, inplace=True)

        # === 不再需要删除元字段，因为从未添加过 ===
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

    # 保存每个文件的独立宽表
    for file_name, wide_df in file_wide_tables.items():
        safe_name = re.sub(r'[^\w]', '_', file_name.replace('.xlsx', ''))
        output_csv = os.path.join(output_dir, f"wide_table_{safe_name}.csv")
        wide_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"✅ 保存文件宽表: {output_csv}")

        feature_dict_df = generate_feature_dictionary(wide_df)
        dict_csv = os.path.join(output_dir, f"feature_dict_{safe_name}.csv")
        feature_dict_df.to_csv(dict_csv, index=False, encoding='utf-8')

    # 合并所有文件宽表（用于建模）
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
