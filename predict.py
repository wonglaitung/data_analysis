import pandas as pd
import numpy as np
import os
from base_model_processor import BaseModelProcessor
import warnings
warnings.filterwarnings('ignore')

class PredictModel(BaseModelProcessor):
        
    def load_models(self):
        """加载训练好的模型"""
        return super().load_models()
    
    def load_feature_config(self):
        """加载特征配置"""
        return super().load_feature_config()
    
    def prepare_predict_data(self, predict_data_path):
        """准备预测数据"""
        try:
            # 加载预测数据
            predict_df = pd.read_csv(predict_data_path)
            print(f"✅ 预测数据已加载: {predict_data_path}, 形状: {predict_df.shape}")
            
            # 确保Id列存在
            if 'Id' not in predict_df.columns:
                print("❌ 预测数据中缺少Id列")
                return None
            
            # 获取预测数据中的特征（不包括Id）
            predict_features = set(predict_df.columns.tolist()) - {'Id'}
            
            # 获取训练时使用的特征
            train_features = set(self.train_feature_names)
            
            # 检查缺少的特征
            missing_features = train_features - predict_features
            
            if missing_features:
                print(f"ℹ️  为与训练数据特征对齐，将填充 {len(missing_features)} 个缺失特征")
                print("   原因：训练时某些类别特征经One-Hot编码后产生了更多特征，预测数据中缺少这些编码后的特征")
                # 为缺少的特征添加默认值0
                for feature in missing_features:
                    predict_df[feature] = 0
            
            # 移除多余的特征
            extra_features = predict_features - train_features
            
            if extra_features:
                print(f"ℹ️  移除 {len(extra_features)} 个训练时未使用的特征")
                print("   原因：预测数据中某些类别特征的取值范围与训练数据不同，产生了额外的One-Hot编码特征")
                predict_df = predict_df.drop(columns=list(extra_features))
            
            # 确保特征顺序与训练时一致
            feature_columns = [col for col in self.train_feature_names if col in predict_df.columns]
            final_columns = ['Id'] + feature_columns
            predict_df = predict_df[final_columns]
            
            print(f"✅ 预测数据准备完成, 最终形状: {predict_df.shape}")
            return predict_df
        except Exception as e:
            print(f"❌ 准备预测数据时出错: {e}")
            return None
    
    def predict(self, predict_df):
        """进行预测"""
        try:
            # 分离Id和特征
            ids = predict_df['Id']
            X = predict_df.drop(columns=['Id'])
            
            # 确保特征列与训练时一致
            # 按照训练时的特征顺序重新排列
            X = X[self.train_feature_names]
            
            # 使用GBDT模型获取叶子节点
            # 使用predict_disable_shape_check=True来忽略特征数量检查
            leaves = self.gbdt_model.booster_.predict(X.values, pred_leaf=True, predict_disable_shape_check=True)
            
            # 对叶子节点进行One-Hot编码，与训练时一致
            n_trees = leaves.shape[1]
            leaves_df_list = []
            
            for i in range(n_trees):
                # 获取第i棵树的叶子节点索引
                tree_leaves = leaves[:, i]
                # 创建DataFrame
                tree_df = pd.DataFrame({f'gbdt_leaf_{i}': tree_leaves})
                # 进行One-Hot编码
                onehot_df = pd.get_dummies(tree_df[f'gbdt_leaf_{i}'], prefix=f'gbdt_leaf_{i}')
                leaves_df_list.append(onehot_df)
            
            # 合并所有树的One-Hot编码结果
            leaves_df = pd.concat(leaves_df_list, axis=1)
            
            # 确保列名与LR模型一致
            # 为缺失的列添加默认值0
            for col in self.lr_feature_names:
                if col not in leaves_df.columns:
                    leaves_df[col] = 0
            
            # 按照LR模型的特征顺序重新排列
            leaves_df = leaves_df[self.lr_feature_names]
            
            # 使用LR模型进行预测
            predictions = self.lr_model.predict_proba(leaves_df)[:, 1]  # 获取正类概率
            
            # 创建结果DataFrame
            result_df = pd.DataFrame({
                'Id': ids,
                'PredictedProb': predictions
            })
            
            return result_df
        except Exception as e:
            print(f"❌ 预测时出错: {e}")
            return None
    
    def run_prediction(self, predict_data_path, output_path="output/prediction_results.csv"):
        """运行完整的预测流程"""
        print("=== 开始预测流程 ===")
        
        # 加载模型
        if not self.load_models():
            return False
        
        # 加载特征配置
        if not self.load_feature_config():
            return False
        
        # 准备预测数据
        predict_df = self.prepare_predict_data(predict_data_path)
        if predict_df is None:
            return False
        
        # 进行预测
        result_df = self.predict(predict_df)
        if result_df is None:
            return False
        
        # 保存结果
        try:
            result_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✅ 预测结果已保存: {output_path}")
            print(f"📊 预测结果统计: 最小值={result_df['PredictedProb'].min():.4f}, 最大值={result_df['PredictedProb'].max():.4f}, 平均值={result_df['PredictedProb'].mean():.4f}")
            return True
        except Exception as e:
            print(f"❌ 保存预测结果时出错: {e}")
            return False

def main():
    # 创建预测模型实例
    predictor = PredictModel()
    
    # 运行预测
    predict_data_path = "output/ml_wide_table_predict_global.csv"
    output_path = "output/prediction_results.csv"
    
    success = predictor.run_prediction(predict_data_path, output_path)
    
    if success:
        print("\n✅ 预测完成!")
    else:
        print("\n❌ 预测失败!")

if __name__ == "__main__":
    main()