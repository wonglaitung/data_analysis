import pandas as pd
import os
from base.base_data_processor import BaseDataProcessor
from base.data_analyzer import DataAnalyzer

class TrainDataProcessor(BaseDataProcessor):
    def __init__(self):
        super().__init__("data_train", "output")
        self.analyzer = DataAnalyzer()
        
    def process_specific_results(self, global_wide, global_dict):
        """
        处理训练数据特有的结果：保存过滤后的特征到config/features.csv
        """
        config_dir = "config"
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, "features.csv")
        global_dict_filtered = global_dict[global_dict['feature_type'] != 'text']
        global_dict_filtered.to_csv(config_file, index=False, encoding='utf-8')
        
        print(f"✅ 全局宽表已保存: {os.path.join(self.output_dir, self.get_global_output_filename())}")
        print(f"✅ 全局字段字典: {os.path.join(self.output_dir, self.get_feature_dict_filename())}")
        print(f"✅ 建模用特征列表: {config_file}")
        
    def get_global_output_filename(self):
        """
        获取训练数据全局输出文件名
        """
        return "ml_wide_table_global.csv"
        
    def get_feature_dict_filename(self):
        """
        获取训练数据特征字典文件名
        """
        return "feature_dictionary_global.csv"

def main(coverage_threshold=0.95, max_top_k=50):
    processor = TrainDataProcessor()
    processor.main(coverage_threshold, max_top_k)

if __name__ == "__main__":
    main(coverage_threshold=0.95, max_top_k=50)