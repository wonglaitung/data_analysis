import pandas as pd
import os
from base_data_processor import BaseDataProcessor

class PredictDataProcessor(BaseDataProcessor):
    def __init__(self):
        super().__init__("data_predict", "output")
        
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
            output_csv = os.path.join(self.output_dir, f"wide_table_predict_{safe_name}.csv")
            wide_df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"✅ 保存文件宽表: {output_csv}")

            feature_dict_df = self.generate_feature_dictionary(wide_df, category_feature_mapping)
            dict_csv = os.path.join(self.output_dir, f"feature_dict_predict_{safe_name}.csv")
            feature_dict_df.to_csv(dict_csv, index=False, encoding='utf-8')

        # 合并所有文件宽表（用于预测）
        print("\n=== 合并所有文件宽表（用于预测）===")
        all_wide_dfs = list(file_wide_tables.values())
        if len(all_wide_dfs) == 1:
            global_wide = all_wide_dfs[0].copy()
        else:
            global_wide = all_wide_dfs[0].copy()
            for df in all_wide_dfs[1:]:
                global_wide = pd.merge(global_wide, df, on='Id', how='outer')

        global_wide = self.calculate_derived_features(global_wide)

        global_output = os.path.join(self.output_dir, "ml_wide_table_predict_global.csv")
        global_wide.to_csv(global_output, index=False, encoding='utf-8')

        global_dict = self.generate_feature_dictionary(global_wide, category_feature_mapping)
        global_dict.to_csv(os.path.join(self.output_dir, "feature_dictionary_predict_global.csv"), index=False, encoding='utf-8')

        # 检查预测数据特征字段与训练时使用的特征字段是否匹配
        self.check_feature_compatibility(global_wide)

        print(f"\n✅ 预测全局宽表已保存: {global_output}")
        print(f"✅ 预测全局字段字典: {os.path.join(self.output_dir, 'feature_dictionary_predict_global.csv')}")
        print(f"\n📊 预测全局宽表最终形状: {global_wide.shape[0]} 行, {global_wide.shape[1]} 列")
        
    def _normalize_name(self, name):
        """
        标准化名称，将特殊字符替换为下划线
        """
        return self.normalize_name(name)

    def check_feature_compatibility(self, predict_wide_df):
        """
        检查预测数据特征字段与训练时使用的特征字段是否匹配
        """
        # 读取训练时使用的特征字段
        train_features_file = os.path.join("config", "features.csv")
        if not os.path.exists(train_features_file):
            print("⚠️  警告: 未找到训练时使用的特征字段文件，跳过兼容性检查")
            return

        try:
            train_features_df = pd.read_csv(train_features_file)
            train_features = set(train_features_df['feature_name'].tolist())
            
            # 获取预测数据的特征字段
            predict_features = set(predict_wide_df.columns.tolist())
            
            # 检查缺少的特征字段
            missing_features = train_features - predict_features
            # 排除Id字段，因为它可能在预测数据中有不同的表示
            if 'Id' in missing_features:
                missing_features.remove('Id')
            
            # 检查多余的特征字段
            extra_features = predict_features - train_features
            # 排除Id字段，因为它可能在预测数据中有不同的表示
            if 'Id' in extra_features:
                extra_features.remove('Id')
            
            if missing_features:
                print(f"⚠️  警告: 预测数据缺少 {len(missing_features)} 个训练时使用的特征字段")
                if len(missing_features) <= 10:
                    print(f"   缺少的特征字段: {', '.join(sorted(missing_features))}")
                else:
                    print(f"   缺少的特征字段 (前10个): {', '.join(sorted(list(missing_features))[:10])}")
            
            if extra_features:
                print(f"ℹ️  提示: 预测数据包含 {len(extra_features)} 个额外的特征字段")
                if len(extra_features) <= 10:
                    print(f"   额外的特征字段: {', '.join(sorted(extra_features))}")
                else:
                    print(f"   额外的特征字段 (前10个): {', '.join(sorted(list(extra_features))[:10])}")
            
            if not missing_features and not extra_features:
                print("✅ 预测数据特征字段与训练时使用的特征字段完全匹配")
            elif not missing_features:
                print("✅ 预测数据包含了训练时使用的所有特征字段")
            else:
                print("❌ 错误: 预测数据缺少训练时使用的特征字段，模型可能无法使用")
                
        except Exception as e:
            print(f"⚠️  警告: 检查特征兼容性时出错: {e}")

def main(coverage_threshold=0.95, max_top_k=50):
    processor = PredictDataProcessor()
    processor.main(coverage_threshold, max_top_k)

if __name__ == "__main__":
    main(coverage_threshold=0.95, max_top_k=50)