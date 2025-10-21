import pandas as pd
import os
import numpy as np
import re
from collections import defaultdict

class DataAnalyzer:
    def __init__(self):
        pass
        
    def analyze_dataset(self, data_dict, file_name=None, output_dir="output", max_combinations=10, use_business_view=True):
        """
        分析数据集
        
        参数:
        data_dict: 包含所有数据的字典，键为文件名，值为DataFrame列表
        file_name: 可选，指定分析单个文件
        output_dir: 输出目录，默认为"output"，如果为None则不保存到Excel文件
        max_combinations: 最大透视表组合数，默认为10
        use_business_view: 是否使用业务视角分析（新方式），默认为True；False时使用原始方式
        
        返回:
        dict: 包含分析结果的字典
        """
        if use_business_view:
            # 使用业务视角分析（新方式）
            return self._analyze_with_business_view(data_dict, file_name, output_dir, max_combinations)
        else:
            # 使用原始分析方式（旧方式）
            return self._analyze_with_original_view(data_dict, file_name)
    
    def _analyze_with_original_view(self, data_dict, file_name=None):
        """
        使用原始方式分析数据集（参考data_analyzer_prev.py）
        
        参数:
        data_dict: 包含所有数据的字典，键为文件名，值为DataFrame列表
        file_name: 可选，指定分析单个文件
        
        返回:
        dict: 包含分析结果的字典
        """
        results = {}
        
        # 如果指定了文件名，则只分析该文件
        if file_name and file_name in data_dict:
            files_to_analyze = {file_name: data_dict[file_name]}
        else:
            files_to_analyze = data_dict
            
        for fname, dataframes in files_to_analyze.items():
            print(f"\n=== 分析文件: {fname} ===")
            file_results = {
                'overview': self._generate_overview(dataframes),
                'field_analysis': self._analyze_fields(dataframes),
                'missing_analysis': self._analyze_missing_values(dataframes)
            }
            results[fname] = file_results
            self._print_analysis_results(fname, file_results)
            
        return results
    
    def _analyze_with_business_view(self, data_dict, file_name=None, output_dir="output", max_combinations=10):
        """
        使用业务视角分析数据集（新方式）
        
        参数:
        data_dict: 包含所有数据的字典，键为文件名，值为DataFrame列表
        file_name: 可选，指定分析单个文件
        output_dir: 输出目录，默认为"output"，如果为None则不保存到Excel文件
        max_combinations: 最大透视表组合数，默认为10
        
        返回:
        dict: 包含分析结果的字典
        """
        results = {}
        
        # 如果output_dir不为None，则创建输出目录
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # 如果指定了文件名，则只分析该文件
        if file_name and file_name in data_dict:
            files_to_analyze = {file_name: data_dict[file_name]}
        else:
            files_to_analyze = data_dict
            
        for fname, dataframes in files_to_analyze.items():
            print(f"\n=== 业务视角数据分析: {fname} ===")
            file_results = {
                'pivot_analysis': self._analyze_pivot_tables(dataframes, fname, max_combinations)
            }
            results[fname] = file_results
            self._print_business_analysis_results(fname, file_results)
            
            # 只有当output_dir不为None时才保存到Excel文件
            if output_dir is not None:
                self._save_business_analysis_to_excel(fname, file_results, output_dir)
            
        return results
    
    def _generate_overview(self, dataframes):
        """
        生成数据概览
        
        参数:
        dataframes: DataFrame列表
        
        返回:
        dict: 数据概览信息
        """
        overview = {
            'total_sheets': len(dataframes),
            'total_rows': sum(len(df) for df in dataframes),
            'total_columns': sum(len(df.columns) for df in dataframes),
            'sheets_info': []
        }
        
        for i, df in enumerate(dataframes):
            sheet_info = {
                'sheet_index': i,
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            }
            overview['sheets_info'].append(sheet_info)
            
        return overview
    
    def _analyze_fields(self, dataframes):
        """
        分析字段信息
        
        参数:
        dataframes: DataFrame列表
        
        返回:
        list: 字段分析结果列表
        """
        field_analysis = []
        
        # 合并所有DataFrame以获取完整的字段信息
        if not dataframes:
            return field_analysis
            
        # 获取所有唯一的列名
        all_columns = set()
        for df in dataframes:
            all_columns.update(df.columns)
            
        # 分析每个字段
        for col in all_columns:
            field_info = {
                'column_name': col,
                'data_types': set(),
                'unique_count': 0,
                'null_count': 0,
                'sample_values': []
            }
            
            # 收集该字段在所有DataFrame中的信息
            all_values = []
            for df in dataframes:
                if col in df.columns:
                    field_info['data_types'].add(str(df[col].dtype))
                    field_info['null_count'] += df[col].isnull().sum()
                    all_values.extend(df[col].dropna().tolist())
                    
            # 计算唯一值数量
            field_info['unique_count'] = len(set(all_values))
            
            # 获取样本值
            field_info['sample_values'] = list(set(all_values))[:10]
            
            field_analysis.append(field_info)
            
        return field_analysis
    
    def _analyze_missing_values(self, dataframes):
        """
        分析缺失值情况
        
        参数:
        dataframes: DataFrame列表
        
        返回:
        list: 缺失值分析结果列表
        """
        missing_analysis = []
        
        # 合并所有DataFrame以获取完整的字段信息
        if not dataframes:
            return missing_analysis
            
        # 获取所有唯一的列名
        all_columns = set()
        for df in dataframes:
            all_columns.update(df.columns)
            
        # 分析每个字段的缺失值情况
        for col in all_columns:
            total_values = 0
            null_count = 0
            
            for df in dataframes:
                if col in df.columns:
                    total_values += len(df)
                    null_count += df[col].isnull().sum()
                    
            missing_percentage = (null_count / total_values * 100) if total_values > 0 else 0
            
            missing_info = {
                'column_name': col,
                'total_values': total_values,
                'null_count': null_count,
                'missing_percentage': missing_percentage
            }
            
            missing_analysis.append(missing_info)
            
        # 按缺失百分比排序
        missing_analysis.sort(key=lambda x: x['missing_percentage'], reverse=True)
        
        return missing_analysis
    
    def _evaluate_categorical_column(self, df, col):
        """
        评估分类字段的价值
        
        参数:
        df: DataFrame
        col: 字段名
        
        返回:
        float: 字段价值评分
        """
        # 获取唯一值数量
        unique_count = df[col].nunique()
        
        # 获取总行数
        total_count = len(df)
        
        # 计算唯一值比例
        unique_ratio = unique_count / total_count if total_count > 0 else 0
        
        # 计算覆盖率（非空值比例）
        coverage = (df[col].notnull().sum()) / total_count if total_count > 0 else 0
        
        # 价值评分：考虑唯一值数量适中且覆盖率高的字段
        # 避免唯一值过多（接近唯一标识符）或过少（无区分度）的字段
        if unique_count <= 1:
            return 0  # 无区分度
        if unique_ratio > 0.9:
            return 0  # 接近唯一标识符
        
        # 综合评分：平衡唯一值数量和覆盖率
        # 使用对数缩放来平衡唯一值数量的影响
        unique_score = np.log(unique_count) if unique_count > 1 else 0
        value_score = unique_score * coverage
        
        return value_score
    
    def _evaluate_numeric_column(self, df, col):
        """
        评估数值字段的价值
        
        参数:
        df: DataFrame
        col: 字段名
        
        返回:
        float: 字段价值评分
        """
        # 检查字段是否真的是数值型（不是分类类型）
        if pd.api.types.is_categorical_dtype(df[col]):
            return 0  # 分类类型不作为数值字段处理
        
        # 检查是否为空
        if df[col].empty or df[col].isnull().all():
            return 0
        
        # 获取总行数
        total_count = len(df)
        
        # 计算覆盖率（非空值比例）
        coverage = (df[col].notnull().sum()) / total_count if total_count > 0 else 0
        
        # 如果覆盖率太低，直接返回0
        if coverage < 0.1:  # 覆盖率低于10%则认为价值低
            return 0
        
        # 检查字段是否真的是数值型
        try:
            # 计算统计指标
            mean_val = df[col].mean()
            std_val = df[col].std()
            max_val = df[col].max()
            min_val = df[col].min()
        except TypeError:
            # 如果计算统计指标失败，说明不是数值型
            return 0
        
        # 计算变异系数（标准差/均值）作为稳定性指标
        cv = (std_val / abs(mean_val)) if mean_val != 0 else 0
        
        # 检查是否有足够的变异性（避免所有值都相同）
        has_variance = std_val > 0 or max_val != min_val
        if not has_variance:
            return 0  # 所有值相同，无分析价值
        
        # 价值评分：考虑覆盖率、变异性和统计特性
        cv_score = 1.0 - abs(cv - 0.5) if 0 <= cv <= 1 else 0.5  # 变异系数适中的字段更有价值
        
        # 考虑字段的平均值和最大值（业务重要性）
        # 使用对数缩放避免极值影响过大
        business_importance = min(np.log10(abs(mean_val) + 1), 15) / 15 if not np.isnan(mean_val) else 0.5
        business_importance = min(business_importance, 1)  # 限制在0-1之间
        
        # 考虑数据分布的多样性
        range_score = min(np.log10(max_val - min_val + 1), 10) / 10 if max_val != min_val else 0
        range_score = min(range_score, 1)  # 限制在0-1之间
        
        # 综合评分
        value_score = coverage * cv_score * 0.4 + business_importance * 0.3 + range_score * 0.3
        
        return value_score
    
    def _select_best_columns(self, df, categorical_columns, numeric_columns, max_categorical=3, max_numeric=3, configured_categorical=None, primary_key_fields=None):
        """
        基于统计特性选择最有价值的字段组合
        
        参数:
        df: DataFrame
        categorical_columns: 分类字段列表
        numeric_columns: 数值字段列表
        max_categorical: 最大分类字段数
        max_numeric: 最大数值字段数
        configured_categorical: 配置文件中指定的类别字段列表
        primary_key_fields: 主键字段列表
        
        返回:
        tuple: (selected_categorical, selected_numeric)
        """
        # 如果没有提供配置的类别字段，则初始化为空列表
        if configured_categorical is None:
            configured_categorical = []
        
        # 如果没有提供主键字段列表，则初始化为空列表
        if primary_key_fields is None:
            primary_key_fields = []
        
        # 从分类字段列表中排除主键字段
        filtered_categorical = [col for col in categorical_columns if col not in primary_key_fields]
        filtered_configured_categorical = [col for col in configured_categorical if col not in primary_key_fields]
        
        # 优先选择配置文件中指定的类别字段（排除主键字段后）
        # 这些字段即使评分不高也会被优先考虑
        priority_categorical = [col for col in filtered_configured_categorical if col in filtered_categorical]
        
        # 评估分类字段价值
        categorical_scores = []
        for col in filtered_categorical:
            score = self._evaluate_categorical_column(df, col)
            categorical_scores.append((col, score))
        
        # 按价值评分排序
        categorical_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 调试信息：打印分类字段评分
        # print("分类字段评分:")
        # for col, score in categorical_scores[:10]:
        #     print(f"  {col}: {score}")
        
        # 选择最有价值的分类字段
        selected_categorical = [col for col, score in categorical_scores[:max_categorical] if score > 0]
        
        # 确保配置文件中指定的类别字段被包含在选择结果中（如果它们有有效评分）
        for col in priority_categorical:
            if col not in selected_categorical and len(selected_categorical) < max_categorical:
                # 检查该字段是否有有效评分
                score = next((s for c, s in categorical_scores if c == col), 0)
                if score > 0:
                    selected_categorical.append(col)
        
        # 如果选择的字段数量不足，从评分高的字段中补充
        if len(selected_categorical) < max_categorical:
            additional_fields = [col for col, score in categorical_scores if col not in selected_categorical and score > 0]
            selected_categorical.extend(additional_fields[:max_categorical - len(selected_categorical)])
        
        # 从数值字段列表中排除主键字段
        filtered_numeric = [col for col in numeric_columns if col not in primary_key_fields]
        
        # 评估数值字段价值
        numeric_scores = []
        for col in filtered_numeric:
            score = self._evaluate_numeric_column(df, col)
            numeric_scores.append((col, score))
        
        # 按价值评分排序
        numeric_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 调试信息：打印数值字段评分
        # print("数值字段评分:")
        # for col, score in numeric_scores[:10]:
        #     print(f"  {col}: {score}")
        # # 打印所有数值字段
        # print("所有数值字段:")
        # for i, col in enumerate(filtered_numeric):
        #     print(f"  {i+1}. {col}")
        
        # 选择最有价值的数值字段
        selected_numeric = [col for col, score in numeric_scores[:max_numeric] if score > 0]
        
        return selected_categorical, selected_numeric
    
    def _analyze_pivot_tables(self, dataframes, file_name=None, max_combinations=10):
        """
        对各字段进行多维度透视分析
        
        参数:
        dataframes: DataFrame列表
        file_name: 文件名（可选）
        
        返回:
        dict: 透视分析结果
        """
        pivot_analysis = {}
        
        # 合并所有DataFrame
        if not dataframes:
            return pivot_analysis
            
        # 合并所有数据帧
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # 如果提供了文件名，读取primary_key.csv配置文件，获取主键字段
        primary_key_fields = []
        if file_name:
            primary_key_file = os.path.join("config", "primary_key.csv")
            if os.path.exists(primary_key_file):
                try:
                    primary_key_df = pd.read_csv(primary_key_file, dtype=str)
                    # 过滤出当前文件的主键配置
                    file_primary_key_df = primary_key_df[primary_key_df['file_name'] == file_name]
                    primary_key_fields = file_primary_key_df['primary_key'].tolist()
                    # 处理带前缀的主键字段名
                    safe_prefix = self._normalize_name(file_name.replace('.xlsx', ''))
                    prefixed_primary_keys = [f"{safe_prefix}_{pk}" for pk in primary_key_fields]
                    primary_key_fields.extend(prefixed_primary_keys)
                    # 同时添加大写和小写形式的主键字段，以处理大小写不一致的情况
                    primary_key_fields_lower = [pk.lower() for pk in primary_key_fields]
                    primary_key_fields_upper = [pk.upper() for pk in primary_key_fields]
                    primary_key_fields.extend(primary_key_fields_lower)
                    primary_key_fields.extend(primary_key_fields_upper)
                    # 调试信息：打印主键字段
                    # print(f"主键字段: {primary_key_fields}")
                except Exception as e:
                    print(f"读取主键配置文件时出错: {e}")
        
        # 获取数值型和分类型字段，排除主键字段
        numeric_columns = [col for col in combined_df.select_dtypes(include=[np.number]).columns.tolist() 
                          if col not in primary_key_fields]
        categorical_columns = [col for col in combined_df.select_dtypes(include=['object', 'category']).columns.tolist() 
                              if col not in primary_key_fields]
        
        # 调试信息：打印部分列名和主键字段
        # print(f"部分列名: {list(combined_df.columns)[:10]}")
        # print(f"主键字段: {primary_key_fields}")
        # print(f"数值型字段: {numeric_columns[:5]}")
        # print(f"分类型字段: {categorical_columns[:5]}")
        
        # 如果提供了文件名，读取category_type.csv配置文件
        category_feature_mapping = {}
        if file_name:
            config_file = os.path.join("config", "category_type.csv")
            if os.path.exists(config_file):
                try:
                    category_df = pd.read_csv(config_file, dtype=str)
                    # 过滤出当前文件的配置
                    file_category_df = category_df[category_df['file_name'] == file_name]
                    for _, row in file_category_df.iterrows():
                        column_name = row['column_name']
                        feature_type = row['feature_type']
                        if feature_type == 'category':
                            # 处理带前缀的列名
                            safe_prefix = self._normalize_name(file_name.replace('.xlsx', ''))
                            prefixed_column = f"{safe_prefix}_{column_name}"
                            # 只添加带前缀的列名（因为实际DataFrame中的列名是带前缀的）
                            category_feature_mapping[prefixed_column] = feature_type
                except Exception as e:
                    print(f"读取配置文件时出错: {e}")
        
        # 将配置文件中指定的类别字段添加到分类字段列表中
        configured_categorical = list(category_feature_mapping.keys())
        # 合并配置的类别字段和自动识别的类别字段，但要排除主键字段
        all_categorical_columns = [col for col in list(set(categorical_columns + configured_categorical)) 
                                  if col not in primary_key_fields]
        
        # 过滤掉不在combined_df中的列名
        all_categorical_columns = [col for col in all_categorical_columns if col in combined_df.columns]
        
        # 确保配置文件中指定的类别字段在DataFrame中被正确设置为category类型
        for col in configured_categorical:
            if col in combined_df.columns and col not in primary_key_fields:
                # 将该列转换为字符串类型，然后再转换为category类型
                combined_df[col] = combined_df[col].astype(str).astype('category')
        
        # 限制字段数量以避免计算量过大
        max_fields = 20
        numeric_columns = numeric_columns[:max_fields]
        all_categorical_columns = all_categorical_columns[:max_fields]
        
        # 基于统计特性选择最有价值的字段
        selected_categorical, selected_numeric = self._select_best_columns(
            combined_df, all_categorical_columns, numeric_columns, max_categorical=3, max_numeric=3, 
            configured_categorical=configured_categorical, primary_key_fields=primary_key_fields
        )
        
        # 对数值型字段进行统计分析
        if selected_numeric:
            pivot_analysis['numeric_summary'] = combined_df[selected_numeric].describe()
        
        # 对分类型字段进行值计数分析
        if selected_categorical:
            pivot_analysis['categorical_summary'] = {}
            for col in selected_categorical:
                value_counts = combined_df[col].value_counts().head(10)  # 只显示前10个值
                pivot_analysis['categorical_summary'][col] = value_counts
                
        # 生成交叉透视表（使用智能选择的字段）
        if len(selected_categorical) >= 2 and len(selected_numeric) >= 1:
            pivot_analysis['pivot_tables'] = []
            
            # 生成多个透视表组合
            combination_count = 0
            
            for i, row_col in enumerate(selected_categorical):
                if combination_count >= max_combinations:
                    break
                for j, col_col in enumerate(selected_categorical):
                    if i >= j or combination_count >= max_combinations:
                        continue
                    for value_col in selected_numeric:
                        if combination_count >= max_combinations:
                            break
                        # 在生成透视表前，确保分类字段被正确处理
                        # 如果行字段或列字段是分类类型，确保它们作为分类处理
                        temp_df = combined_df.copy()
                        
                        # 确保行字段和列字段作为字符串处理，避免数值聚合
                        if row_col in temp_df.columns:
                            temp_df[row_col] = temp_df[row_col].astype(str)
                        if col_col in temp_df.columns:
                            temp_df[col_col] = temp_df[col_col].astype(str)
                        
                        pivot_table = pd.pivot_table(
                            temp_df,
                            values=value_col,
                            index=row_col,
                            columns=col_col,
                            aggfunc='sum',
                            fill_value=0
                        )
                        pivot_analysis['pivot_tables'].append({
                            'row_field': row_col,
                            'col_field': col_col,
                            'value_field': value_col,
                            'pivot_table': pivot_table.head(20)  # 限制显示20行
                        })
                        combination_count += 1
                        if combination_count >= max_combinations:
                            break
                if combination_count >= max_combinations:
                    break
        
        return pivot_analysis
    
    def _print_business_analysis_results(self, file_name, analysis):
        """
        从商业视角打印分析结果到屏幕
        
        参数:
        file_name: 文件名
        analysis: 分析结果字典
        """
        print(f"\n--- {file_name} 业务洞察分析 ---")
        pivot_analysis = analysis['pivot_analysis']
        
        if 'pivot_tables' in pivot_analysis and pivot_analysis['pivot_tables']:
            print("关键业务指标透视分析:")
            for i, pivot_info in enumerate(pivot_analysis['pivot_tables'], 1):
                print(f"\n{i}. {pivot_info['value_field']} 按 {pivot_info['row_field']} 和 {pivot_info['col_field']} 的分布:")
                print(pivot_info['pivot_table'])
                print()
        else:
            print("未能生成有效的业务透视分析表")
            
        # 提供业务洞察建议
        print("--- 业务洞察建议 ---")
        print("1. 请关注不同维度组合下的关键指标分布情况")
        print("2. 寻找异常值或特殊模式，这些可能代表业务机会或风险")
        print("3. 对比不同时间段或不同类别的数据，识别趋势和差异")
    
    def _print_analysis_results(self, file_name, analysis):
        """
        打印分析结果到屏幕（原有方式）
        
        参数:
        file_name: 文件名
        analysis: 分析结果字典
        """
        print(f"\n--- {file_name} 数据概览 ---")
        overview = analysis['overview']
        print(f"总工作表数: {overview['total_sheets']}")
        print(f"总行数: {overview['total_rows']}")
        print(f"总列数: {overview['total_columns']}")
        
        print(f"\n--- {file_name} 字段分析 ---")
        field_analysis = analysis['field_analysis']
        for field in field_analysis[:10]:  # 只显示前10个字段
            print(f"字段名: {field['column_name']}")
            print(f"  数据类型: {', '.join(field['data_types'])}")
            print(f"  唯一值数量: {field['unique_count']}")
            print(f"  空值数量: {field['null_count']}")
            print(f"  样本值: {field['sample_values']}")
            print()
        
        if len(field_analysis) > 10:
            print(f"  ... 还有 {len(field_analysis) - 10} 个字段未显示")
        
        print(f"\n--- {file_name} 缺失值分析 ---")
        missing_analysis = analysis['missing_analysis']
        for missing in missing_analysis[:10]:  # 只显示前10个字段
            print(f"字段名: {missing['column_name']}")
            print(f"  总值数: {missing['total_values']}")
            print(f"  空值数: {missing['null_count']}")
            print(f"  缺失百分比: {missing['missing_percentage']:.2f}%")
            print()
        
        if len(missing_analysis) > 10:
            print(f"  ... 还有 {len(missing_analysis) - 10} 个字段未显示")
    
    def _save_business_analysis_to_excel(self, file_name, analysis, output_dir="output"):
        """
        将业务分析结果保存为Excel文件
        
        参数:
        file_name: 文件名
        analysis: 分析结果字典
        output_dir: 输出目录
        """
        import matplotlib.pyplot as plt
        from openpyxl.drawing.image import Image
        from openpyxl.chart import BarChart, LineChart, Reference
        from openpyxl.chart.series import DataPoint
        import io
        import base64
        
        # 创建Excel写入器
        safe_file_name = self._normalize_name(file_name)
        excel_path = os.path.join(output_dir, f"{safe_file_name}_business_analysis.xlsx")
        writer = pd.ExcelWriter(excel_path, engine='openpyxl')
        
        pivot_analysis = analysis['pivot_analysis']
        
        # 保存透视表到不同的工作表
        if 'pivot_tables' in pivot_analysis and pivot_analysis['pivot_tables']:
            for i, pivot_info in enumerate(pivot_analysis['pivot_tables'], 1):
                # 创建有意义的工作表名称
                value_field = pivot_info['value_field']
                row_field = pivot_info['row_field']
                col_field = pivot_info['col_field']
                
                # 限制字段名称长度，避免工作表名称过长
                value_field_short = value_field[:15] if len(value_field) > 15 else value_field
                row_field_short = row_field[:10] if len(row_field) > 10 else row_field
                col_field_short = col_field[:10] if len(col_field) > 10 else col_field
                
                # 创建工作表名称
                sheet_name = f"{value_field_short}_{row_field_short}_{col_field_short}"
                # 确保工作表名称符合Excel的命名规则（不超过31个字符）
                sheet_name = sheet_name[:31]
                # 移除可能引起问题的特殊字符
                sheet_name = re.sub(r'[\/*?:\[\]]', '_', sheet_name)
                
                pivot_table = pivot_info['pivot_table']
                # 重置索引以便保存
                pivot_df = pivot_table.reset_index()
                pivot_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # 添加表头信息
                worksheet = writer.sheets[sheet_name]
                worksheet.cell(row=1, column=1, value=f"{pivot_info['value_field']} 按 {pivot_info['row_field']} 和 {pivot_info['col_field']} 的分布")
                
                # 为每个透视表工作表创建图表
                chart = BarChart()
                chart.title = f"{pivot_info['value_field']} 按 {pivot_info['row_field']} 的分布"
                chart.x_axis.title = pivot_info['row_field']
                chart.y_axis.title = pivot_info['value_field']
                
                # 创建数据引用
                data_start_col = 2  # B列开始（A列是行标签）
                data_end_col = len(pivot_df.columns) - 1
                data_start_row = 2  # 第2行开始（第1行是标题）
                data_end_row = len(pivot_df) + 1  # 加1是因为第一行是标题
                
                # 设置数据和标签
                cats = Reference(worksheet, min_col=1, min_row=data_start_row, max_row=data_end_row)
                data = Reference(worksheet, min_col=data_start_col, min_row=data_start_row - 1, max_row=data_end_row, max_col=data_end_col + 1)
                
                chart.add_data(data, titles_from_data=True)
                chart.set_categories(cats)
                
                # 调整图表大小
                chart.width = 20
                chart.height = 10
                
                # 将图表添加到工作表
                worksheet.add_chart(chart, "A15")  # 在A15位置开始放置图表

        # 创建一个单独的图表工作表
        if 'pivot_tables' in pivot_analysis and pivot_analysis['pivot_tables']:
            chart_sheet_name = "可视化图表"
            writer.book.create_sheet(chart_sheet_name)
            chart_worksheet = writer.book[chart_sheet_name]
            
            # 添加描述
            chart_worksheet['A1'] = "业务数据可视化图表"
            
            # 针对第一个透视表数据创建可视化
            first_pivot = pivot_analysis['pivot_tables'][0]['pivot_table']
            first_pivot_info = pivot_analysis['pivot_tables'][0]
            
            # 创建图表
            chart = LineChart()
            chart.title = f"{first_pivot_info['value_field']} 趋势分析"
            chart.x_axis.title = first_pivot_info['row_field']
            chart.y_axis.title = first_pivot_info['value_field']
            
            # 将透视表数据转换为适合图表的格式
            chart_df = first_pivot.reset_index()
            chart_df.to_excel(writer, sheet_name=chart_sheet_name, startrow=2, index=False)
            
            # 创建数据引用
            data_start_row = 3  # 第3行开始（第1行标题，第2行是Excel标题）
            data_end_row = len(chart_df) + 2  # 加2是因为第一行是标题，Excel索引从1开始
            data_start_col = 2  # B列开始（A列是行标签）
            data_end_col = len(chart_df.columns)  # 包含所有列
            
            # 为图表设置数据
            cats = Reference(chart_worksheet, min_col=1, min_row=data_start_row, max_row=data_end_row)
            data = Reference(chart_worksheet, min_col=data_start_col, min_row=data_start_row - 1, max_row=data_end_row, max_col=data_end_col)
            
            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)
            
            # 调整图表大小
            chart.width = 25
            chart.height = 12
            
            # 将图表添加到图表工作表
            chart_worksheet.add_chart(chart, "A20")
        
        # 保存业务洞察建议到单独的工作表
        insights_df = pd.DataFrame({
            '业务洞察建议': [
                "1. 请关注不同维度组合下的关键指标分布情况",
                "2. 寻找异常值或特殊模式，这些可能代表业务机会或风险",
                "3. 对比不同时间段或不同类别的数据，识别趋势和差异"
            ]
        })
        insights_df.to_excel(writer, sheet_name='业务洞察建议', index=False)
        
        # 保存分析摘要到单独的工作表
        if 'pivot_tables' in pivot_analysis and pivot_analysis['pivot_tables']:
            summary_data = []
            for i, pivot_info in enumerate(pivot_analysis['pivot_tables'], 1):
                summary_data.append({
                    '序号': i,
                    '指标名称': pivot_info['value_field'],
                    '行维度': pivot_info['row_field'],
                    '列维度': pivot_info['col_field'],
                    '数据行数': len(pivot_info['pivot_table']),
                    '数据列数': len(pivot_info['pivot_table'].columns)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='分析摘要', index=False)
        
        # 关闭写入器
        writer.close()
        print(f"✅ 业务分析结果已保存到: {excel_path}")
    
    def _normalize_name(self, name):
        """
        标准化名称，将特殊字符替换为下划线
        """
        import re
        return re.sub(r'[^\w]', '_', name)