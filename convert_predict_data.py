import pandas as pd
import os
from base.base_data_processor import BaseDataProcessor
from base.data_analyzer import DataAnalyzer

class PredictDataProcessor(BaseDataProcessor):
    def __init__(self):
        super().__init__("data_predict", "output")
        self.analyzer = DataAnalyzer()
        
    def process_specific_results(self, global_wide, global_dict):
        """
        处理预测数据特有的结果：检查特征兼容性
        """
        # 检查预测数据特征字段与训练时使用的特征字段是否匹配
        self.check_feature_compatibility(global_wide)
        
        print(f"✅ 预测全局宽表已保存: {os.path.join(self.output_dir, self.get_global_output_filename())}")
        print(f"✅ 预测全局字段字典: {os.path.join(self.output_dir, self.get_feature_dict_filename())}")
        
    def get_global_output_filename(self):
        """
        获取预测数据全局输出文件名
        """
        return "ml_wide_table_predict_global.csv"
        
    def get_feature_dict_filename(self):
        """
        获取预测数据特征字典文件名
        """
        return "feature_dictionary_predict_global.csv"

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