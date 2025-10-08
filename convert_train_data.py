import pandas as pd
import os
from base.base_data_processor import BaseDataProcessor

class TrainDataProcessor(BaseDataProcessor):
    def __init__(self):
        super().__init__("data_train", "output")
        
    def main(self, coverage_threshold=0.95, max_top_k=50):
        # 获取类别特征映射
        all_data, all_primary_keys, category_feature_mapping = self.process_all_excel_files()
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

        # 保存每个文件的独立宽表
        for file_name, wide_df in file_wide_tables.items():
            safe_name = self._normalize_name(file_name.replace('.xlsx', ''))
            output_csv = os.path.join(self.output_dir, f"wide_table_{safe_name}.csv")
            wide_df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"✅ 保存文件宽表: {output_csv}")

            feature_dict_df = self.generate_feature_dictionary(wide_df, category_feature_mapping)
            dict_csv = os.path.join(self.output_dir, f"feature_dict_{safe_name}.csv")
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