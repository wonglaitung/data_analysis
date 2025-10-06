import os
#os.environ["NUMBA_DISABLE_TBB"] = "1"
import pandas as pd
import numpy as np
from base_model_processor import BaseModelProcessor
import warnings
import joblib
import logging
import lightgbm as lgb
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

# ========== 加载模型和元数据 ==========
MODEL_DIR = 'output'

def load_models():
    required_files = [
        'gbdt_model.pkl',
        'lr_model.pkl',
        'train_feature_names.csv',
        'category_features.csv',
        'continuous_features.csv',
        'actual_n_estimators.csv'
    ]
    model_dir = Path(MODEL_DIR)
    for f in required_files:
        if not (model_dir / f).exists():
            raise FileNotFoundError(f"❌ 找不到必需文件: {model_dir / f}")

    gbdt_model = joblib.load(model_dir / 'gbdt_model.pkl')
    lr_model = joblib.load(model_dir / 'lr_model.pkl')
    train_feature_names = pd.read_csv(model_dir / 'train_feature_names.csv')['feature'].tolist()
    category_features = pd.read_csv(model_dir / 'category_features.csv')['feature'].tolist()
    continuous_features = pd.read_csv(model_dir / 'continuous_features.csv')['feature'].tolist()
    actual_n_estimators = pd.read_csv(model_dir / 'actual_n_estimators.csv')['n_estimators'].iloc[0]
    category_prefixes = [col + "_" for col in category_features]

    logging.info(f"✅ 模型加载完成，实际树数量: {actual_n_estimators}")
    return {
        'gbdt_model': gbdt_model,
        'lr_model': lr_model,
        'train_feature_names': train_feature_names,
        'category_features': category_features,
        'continuous_features': continuous_features,
        'actual_n_estimators': actual_n_estimators,
        'category_prefixes': category_prefixes
    }


# ========== 工具函数：解析叶子路径 ==========
def get_leaf_path_enhanced(booster, tree_index, leaf_index, feature_names, category_prefixes):
    try:
        model_dump = booster.dump_model()
        if tree_index >= len(model_dump['tree_info']):
            return None
        tree_info = model_dump['tree_info'][tree_index]['tree_structure']
    except Exception as e:
        logging.warning(f"解析树结构失败: {e}")
        return None

    node_stack = [(tree_info, [])]
    while node_stack:
        node, current_path = node_stack.pop()
        if 'leaf_index' in node and node['leaf_index'] == leaf_index:
            return current_path
        if 'split_feature' in node:
            feat_idx = node['split_feature']
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature_{feat_idx}"
            threshold = node.get('threshold', 0.0)
            decision_type = node.get('decision_type', '<=')

            is_category = False
            original_col = None
            category_value = None
            for prefix in category_prefixes:
                if feat_name.startswith(prefix):
                    is_category = True
                    original_col = prefix.rstrip('_')
                    category_value = feat_name[len(prefix):]
                    break

            if is_category:
                right_rule = f"{original_col} == '{category_value}'"
                left_rule = f"{original_col} != '{category_value}'"
            else:
                if decision_type in ('<=', 'no_greater'):
                    right_rule = f"{feat_name} > {threshold:.4f}"
                    left_rule = f"{feat_name} <= {threshold:.4f}"
                else:
                    right_rule = f"{feat_name} {decision_type} {threshold:.4f}"
                    left_rule = f"{feat_name} not {decision_type} {threshold:.4f}"

            if 'right_child' in node:
                node_stack.append((node['right_child'], current_path + [right_rule]))
            if 'left_child' in node:
                node_stack.append((node['left_child'], current_path + [left_rule]))
    return None


# ========== 预处理单样本 ==========
def preprocess_single_sample(sample_dict, continuous_features, category_features, train_feature_names):
    sample_df = pd.DataFrame([sample_dict])
    # 连续特征
    for col in continuous_features:
        if col not in sample_df.columns:
            sample_df[col] = -1
        else:
            sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce').fillna(-1)
    # 分类特征
    all_dummies_list = []
    for col in category_features:
        if col in sample_df.columns:
            val = sample_df[col].iloc[0]
            sample_df[col] = "-1" if pd.isna(val) or val == "" else str(val)
        else:
            sample_df[col] = "-1"
        sample_df[col] = sample_df[col].astype('category')
        dummies = pd.get_dummies(sample_df[col], prefix=col)
        # 补齐训练时的 dummy 列
        missing_cols = [train_col for train_col in train_feature_names if train_col.startswith(col + "_") and train_col not in dummies.columns]
        if missing_cols:
            missing_df = pd.DataFrame(0, index=dummies.index, columns=missing_cols)
            dummies = pd.concat([dummies, missing_df], axis=1)
        all_dummies_list.append(dummies)
    # 合并
    if all_dummies_list:
        dummies_combined = pd.concat(all_dummies_list, axis=1)
        # 移除dummies_combined中的重复列
        dummies_combined = dummies_combined.loc[:, ~dummies_combined.columns.duplicated()]
        sample_df = pd.concat([sample_df.drop(columns=category_features), dummies_combined], axis=1)
    else:
        sample_df = sample_df.drop(columns=category_features)
    # 补齐所有训练特征
    missing_final_cols = set(train_feature_names) - set(sample_df.columns)
    if missing_final_cols:
        missing_final_df = pd.DataFrame(0, index=sample_df.index, columns=list(missing_final_cols))
        sample_df = pd.concat([sample_df, missing_final_df], axis=1)
    # 移除sample_df中的重复列
    sample_df = sample_df.loc[:, ~sample_df.columns.duplicated()]
    return sample_df.reindex(columns=train_feature_names, fill_value=0)


# ========== 核心预测函数 ==========
def predict_core(sample_df_list, models, return_explanation=True, generate_plot=False, calculate_shap=False):
    """
    与 app.py 中的 predict_core 完全一致
    """
    if not sample_df_list:
        return []

    from sklearn.linear_model import LogisticRegression
    import numpy as np

    batch_df = pd.concat(sample_df_list, ignore_index=True)
    gbdt_model = models['gbdt_model']
    lr_model = models['lr_model']
    train_feature_names = models['train_feature_names']
    actual_n_estimators = models['actual_n_estimators']
    category_prefixes = models['category_prefixes']

    # Step 1: GBDT 叶子索引
    leaf_indices_batch = gbdt_model.booster_.predict(batch_df.values, pred_leaf=True)
    n_trees = actual_n_estimators

    # Step 2: 叶子 One-Hot
    leaf_dummies_list = []
    for i in range(n_trees):
        leaf_col_name = f"gbdt_leaf_{i}"
        leaf_series = pd.Series(leaf_indices_batch[:, i], name=leaf_col_name)
        dummies = pd.get_dummies(leaf_series, prefix=leaf_col_name)
        leaf_dummies_list.append(dummies)
    leaf_dummies_combined = pd.concat(leaf_dummies_list, axis=1) if leaf_dummies_list else pd.DataFrame()

    lr_feature_names = getattr(lr_model, 'feature_names_in_', [f"feature_{i}" for i in range(len(lr_model.coef_[0]))])
    missing_leaf_cols = set(lr_feature_names) - set(leaf_dummies_combined.columns)
    if missing_leaf_cols:
        missing_leaf_df = pd.DataFrame(0, index=leaf_dummies_combined.index, columns=list(missing_leaf_cols))
        leaf_dummies_combined = pd.concat([leaf_dummies_combined, missing_leaf_df], axis=1)
    leaf_dummies_combined = leaf_dummies_combined.reindex(columns=lr_feature_names, fill_value=0)

    # Step 3: LR 概率
    probabilities = lr_model.predict_proba(leaf_dummies_combined)[:, 1]

    if not return_explanation or not calculate_shap:
        # 如果不返回解释或不计算SHAP值，直接返回概率
        return [{"probability": round(float(p), 4), "explanation": None} for p in probabilities]

    # Step 4: 特征贡献解释（使用LightGBM内置功能替代SHAP）
    contrib_values_batch = None
    if calculate_shap:
        try:
            # 使用LightGBM内置的pred_contrib功能计算特征贡献
            contrib_values_batch = gbdt_model.booster_.predict(batch_df.values, pred_contrib=True)
            # contrib_values_batch的形状为 (n_samples, n_features + 1)
            # 最后一列是期望值（base value），前面的列是各特征的贡献值
        except Exception as e:
            logging.error(f"特征贡献计算失败: {e}")
            # 返回空解释
            return [{
                "probability": round(float(probabilities[i]), 4),
                "explanation": {
                    "important_features": [],
                    "shap_plot_base64": "",
                    "top_rules": [],
                    "feature_based_rules": []
                }
            } for i in range(len(sample_df_list))]
    else:
        # 如果不计算特征贡献，返回空解释
        return [{
            "probability": round(float(probabilities[i]), 4),
            "explanation": {
                "important_features": [],
                "shap_plot_base64": "",
                "top_rules": [],
                "feature_based_rules": []
            }
        } for i in range(len(sample_df_list))]

    results = []
    for idx in range(len(sample_df_list)):
        # 获取当前样本的特征贡献值（排除最后的期望值列）
        contrib_vals = contrib_values_batch[idx, :-1]
        feature_imp = [(train_feature_names[i], float(contrib_vals[i])) for i in range(len(contrib_vals))]
        feature_imp.sort(key=lambda x: abs(x[1]), reverse=True)
        important_features = [{"feature": feat, "shap_value": round(val, 4)} for feat, val in feature_imp[:3]]
        top_contrib_features = [feat for feat, val in feature_imp[:5]]
        leaf_indices = leaf_indices_batch[idx]

        # 原始路径规则
        path_rules = []
        for tree_idx in range(min(3, len(leaf_indices))):
            leaf_idx = leaf_indices[tree_idx]
            rule = get_leaf_path_enhanced(gbdt_model.booster_, tree_idx, leaf_idx, train_feature_names, category_prefixes)
            if rule:
                path_rules.extend(rule[:3])
        seen = set()
        unique_path_rules = []
        for r in path_rules:
            if r not in seen:
                seen.add(r)
                unique_path_rules.append(r)
        top_rules = unique_path_rules[:5]

        # 特征关联规则
        feature_rules = []
        for tree_idx in range(min(10, len(leaf_indices))):
            leaf_idx = leaf_indices[tree_idx]
            rule = get_leaf_path_enhanced(gbdt_model.booster_, tree_idx, leaf_idx, train_feature_names, category_prefixes)
            if rule:
                for r in rule:
                    for feat in top_contrib_features:
                        if feat in r or (feat.split('_')[0] + " " in r) or (feat.split('_')[0] + " ==" in r):
                            contrib_val = next((val for f, val in feature_imp if f == feat), 0)
                            rule_with_contrib = f"{r} (贡献值: {contrib_val:+.4f})"
                            if rule_with_contrib not in feature_rules:
                                feature_rules.append(rule_with_contrib)
                            break
                    if len(feature_rules) >= 5:
                        break
            if len(feature_rules) >= 5:
                break

        # 生成图（仅第一个样本）
        # 由于不再使用SHAP，此部分将生成一个简单的特征贡献图
        shap_plot_b64 = ""
        if generate_plot and idx == 0:
            try:
                import matplotlib.pyplot as plt
                import base64
                import io
                
                # 获取前10个最重要的特征及其贡献值
                top_features_plot = feature_imp[:10]
                features_plot = [feat for feat, _ in top_features_plot]
                contribs_plot = [val for _, val in top_features_plot]
                
                # 创建水平条形图
                plt.figure(figsize=(10, 6))
                y_pos = range(len(features_plot))
                colors = ['green' if x > 0 else 'red' for x in contribs_plot]
                plt.barh(y_pos, contribs_plot, color=colors)
                plt.yticks(y_pos, features_plot)
                plt.xlabel('Feature Contribution')
                plt.title('Top 10 Feature Contributions for Prediction')
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                buf.seek(0)
                shap_plot_b64 = "image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except Exception as e:
                logging.warning(f"特征贡献图生成失败: {e}")

        explanation = {
            "important_features": important_features,
            "shap_plot_base64": shap_plot_b64,
            "top_rules": top_rules,
            "feature_based_rules": feature_rules[:5]
        }

        results.append({
            "probability": round(float(probabilities[idx]), 4),
            "explanation": explanation
        })

    return results


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
    
    def predict_with_explanation(self, predict_df, calculate_shap=False):
        """进行预测并生成解释性信息"""
        try:
            # 加载模型和元数据
            models = load_models()
            
            # 预处理数据
            processed_rows = []
            for _, row in predict_df.iterrows():
                processed = preprocess_single_sample(
                    row.to_dict(),
                    models['continuous_features'],
                    models['category_features'],
                    models['train_feature_names']
                )
                # 修复可能的重复列名问题
                processed = processed.loc[:, ~processed.columns.duplicated()]
                processed_rows.append(processed)
            
            # 检查processed_rows中的DataFrame是否有重复列名
            for i, df in enumerate(processed_rows):
                if df.columns.duplicated().any():
                    print(f"第{i}个样本存在重复列名")
                    duplicated_cols = df.columns[df.columns.duplicated()]
                    print(f"重复的列名: {duplicated_cols}")
                    # 移除重复列，保留第一个
                    processed_rows[i] = df.loc[:, ~df.columns.duplicated()]
            
            # 使用predict_core进行预测，返回解释性信息
            results = predict_core(processed_rows, models, return_explanation=True, generate_plot=False, calculate_shap=calculate_shap)
            
            # 构造 CSV 结果（完全复刻 app.py）
            csv_results = []
            for r in results:
                exp = r["explanation"]
                # 检查exp是否为None
                if exp is None:
                    top_features = ""
                    top_rules = ""
                else:
                    top_features = "; ".join([f"{feat['feature']}({feat['shap_value']:+.3f})" for feat in exp["important_features"]]) if exp.get("important_features") else ""
                    top_rules = "; ".join(exp["top_rules"][:3]) if exp.get("top_rules") else ""
                csv_results.append({"probability": r["probability"], "top_features": top_features, "top_rules": top_rules})

            # 生成结果 DataFrame
            result_df = pd.DataFrame()
            result_df['Id'] = predict_df['Id'].values
            result_df['PredictedProb'] = [r['probability'] for r in csv_results]
            result_df['top_features'] = [r['top_features'] for r in csv_results]
            result_df['top_rules'] = [r['top_rules'] for r in csv_results]
            
            return result_df
        except Exception as e:
            print(f"❌ 预测时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_prediction(self, predict_data_path, output_path="output/prediction_results.csv", calculate_shap=False):
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
        
        # 进行预测并生成解释性信息
        result_df = self.predict_with_explanation(predict_df, calculate_shap)
        if result_df is None:
            return False
        
        # 重命名列以匹配local_batch_predict.py的输出格式
        # 注意：我们保留原始列名，不进行重命名，以确保与输出文件格式一致
        # result_df = result_df.rename(columns={
        #     'PredictedProb': 'prediction_probability',
        #     'top_features': 'top_3_features',
        #     'top_rules': 'top_3_rules'
        # })
        
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
    # 检查是否有--shap参数
    calculate_shap = "--shap" in sys.argv
    
    # 创建预测模型实例
    predictor = PredictModel()
    
    # 运行预测
    predict_data_path = "output/ml_wide_table_predict_global.csv"
    output_path = "output/prediction_results.csv"
    
    success = predictor.run_prediction(predict_data_path, output_path, calculate_shap)
    
    if success:
        print("\n✅ 预测完成!")
        if calculate_shap:
            print("✅ SHAP值已计算并包含在结果中")
        else:
            print("ℹ️  仅进行预测，未计算SHAP值（使用--shap参数可启用SHAP计算）")
    else:
        print("\n❌ 预测失败!")

if __name__ == "__main__":
    main()
