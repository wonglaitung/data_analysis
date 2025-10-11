import pandas as pd
import os
from base.base_data_processor import BaseDataProcessor
from base.data_analyzer import DataAnalyzer

class TrainDataProcessor(BaseDataProcessor):
    def __init__(self):
        super().__init__("data_train", "output")
        self.analyzer = DataAnalyzer()
        
    def main(self, coverage_threshold=0.95, max_top_k=50):
        # 获取类别特征映射
        all_data, all_primary_keys, category_feature_mapping = self.process_all_excel_files()
        
        # 进行数据分析
        print("\n=== 开始数据分析 ===")
        analysis_results = self.analyzer.analyze_dataset(all_data)
        print("✅ 数据分析完成")
        
        field_analysis, dimension_analysis = self.analyze_fields_and_dimensions(all_data, category_feature_mapping)
        print(f"\n分析了 {len(field_analysis)} 个字段, {len(dimension_analysis)} 个维度")

        file_wide_tables = self.create_wide_table_per_file(
            all_data,
            dimension_analysis,
            all_primary_keys,
            category_feature_mapping,
            coverage_threshold=coverage_threshold,
            max_top_k=max_top_k
        )

        if not file_wide_tables:
            print("❌ 未生成任何宽表")
            return

        # 不再保存每个文件的独立宽表，只保留最终大宽表
        # 生成每个文件的特征字典但不保存
        for file_name, wide_df in file_wide_tables.items():
            feature_dict_df = self.generate_feature_dictionary(wide_df, category_feature_mapping)

        # 合并所有文件宽表（用于建模）
        print("\n=== 合并所有文件宽表（用于建模）===")
        all_wide_dfs = list(file_wide_tables.values())
        if len(all_wide_dfs) == 1:
            global_wide = all_wide_dfs[0].copy()
            print(f"  只有一个文件宽表，匹配率: 100%")
        else:
            global_wide = all_wide_dfs[0].copy()
            total_ids = len(global_wide)
            for df in all_wide_dfs[1:]:
                matched_ids = len(pd.merge(global_wide[['Id']], df[['Id']], on='Id', how='inner'))
                match_rate = matched_ids / total_ids if total_ids > 0 else 0
                print(f"  与 {df.shape[0]} 行的宽表合并，基于主键(Id)匹配率: {match_rate:.2%}")
                global_wide = pd.merge(global_wide, df, on='Id', how='outer')
                total_ids = len(global_wide)

        global_wide = self.calculate_derived_features(global_wide)

        global_output = os.path.join(self.output_dir, "ml_wide_table_global.csv")
        global_wide.to_csv(global_output, index=False, encoding='utf-8')

        global_dict = self.generate_feature_dictionary(global_wide, category_feature_mapping)
        global_dict.to_csv(os.path.join(self.output_dir, "feature_dictionary_global.csv"), index=False, encoding='utf-8')

        config_dir = "config"
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, "features.csv")
        global_dict_filtered = global_dict[global_dict['feature_type'] != 'text']
        global_dict_filtered.to_csv(config_file, index=False, encoding='utf-8')

        print(f"\n✅ 全局宽表已保存: {global_output}")
        print(f"✅ 全局字段字典: {os.path.join(self.output_dir, 'feature_dictionary_global.csv')}")
        print(f"✅ 建模用特征列表: {config_file}")
        print(f"\n📊 全局宽表最终形状: {global_wide.shape[0]} 行, {global_wide.shape[1]} 列")
        
    def _normalize_name(self, name):
        """
        标准化名称，将特殊字符替换为下划线
        """
        return self.normalize_name(name)

def main(coverage_threshold=0.95, max_top_k=50):
    processor = TrainDataProcessor()
    processor.main(coverage_threshold, max_top_k)

if __name__ == "__main__":
    main(coverage_threshold=0.95, max_top_k=50)