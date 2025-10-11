import pandas as pd
import os
import numpy as np
from collections import defaultdict

class DataAnalyzer:
    def __init__(self):
        pass
        
    def analyze_dataset(self, data_dict, file_name=None):
        """
        分析数据集，生成数据概览、字段分析和缺失值分析
        
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
    
    def _print_analysis_results(self, file_name, analysis):
        """
        打印分析结果到屏幕
        
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
    
    def _normalize_name(self, name):
        """
        标准化名称，将特殊字符替换为下划线
        """
        import re
        return re.sub(r'[^\w]', '_', name)