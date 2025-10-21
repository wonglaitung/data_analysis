import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from base.data_analyzer import DataAnalyzer

def analyze_single_excel_file(file_path, coverage_threshold=0.95, max_top_k=50, max_combinations=10):
    """
    对单个Excel文件进行数据分析
    
    参数:
    file_path: Excel文件路径
    coverage_threshold: 覆盖率阈值
    max_top_k: 最大top-k值
    max_combinations: 最大透视表组合数
    """
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        return
    
    print(f"正在分析文件: {file_path}")
    
    try:
        # 读取Excel文件的所有工作表
        excel_data = pd.read_excel(file_path, sheet_name=None)
        
        # 获取文件名
        file_name = os.path.basename(file_path)
        
        # 读取primary_key.csv配置文件，获取主键字段列表
        primary_key_fields = []
        primary_key_file = os.path.join("config", "primary_key.csv")
        if os.path.exists(primary_key_file):
            try:
                primary_key_df = pd.read_csv(primary_key_file, dtype=str)
                # 过滤出当前文件的主键配置
                file_primary_key_df = primary_key_df[primary_key_df['file_name'] == file_name]
                primary_key_fields = file_primary_key_df['primary_key'].tolist()
            except Exception as e:
                print(f"读取主键配置文件时出错: {e}")
        
        # 在读取后立即处理配置为category类型的字段
        # 读取category_type.csv配置文件
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
                        # 处理每个工作表中的该列
                        for sheet_name, df in excel_data.items():
                            if column_name in df.columns:
                                # 将该列转换为字符串类型
                                df[column_name] = df[column_name].astype(str)
            except Exception as e:
                print(f"读取配置文件时出错: {e}")
        
        # 准备数据字典
        data_dict = {file_name: list(excel_data.values())}
        
        # 创建数据分析器实例
        analyzer = DataAnalyzer()
        
        # 执行数据分析，传递max_combinations参数
        analysis_results = analyzer.analyze_dataset(data_dict, file_name, max_combinations=max_combinations)
        
        print(f"\n✅ 文件 {file_name} 分析完成")
        return analysis_results
        
    except Exception as e:
        print(f"分析文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='分析Excel文件')
    parser.add_argument('file_path', help='Excel文件路径')
    parser.add_argument('--max-combinations', type=int, default=10, help='最大透视表组合数，默认为10')
    
    # 如果没有提供参数，显示帮助信息
    if len(sys.argv) < 2:
        parser.print_help()
        return
    
    # 获取命令行参数
    args = parser.parse_args()
    file_path = args.file_path
    max_combinations = args.max_combinations
    
    # 执行数据分析
    analyze_single_excel_file(file_path, max_combinations=max_combinations)

if __name__ == "__main__":
    main()