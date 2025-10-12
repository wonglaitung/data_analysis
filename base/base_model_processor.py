import pandas as pd
import numpy as np
import os
import joblib
import warnings
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
import matplotlib.pyplot as plt
import platform

# 深度学习相关导入检查
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    print("警告: 未安装PyTorch，将跳过深度学习模型相关功能")

warnings.filterwarnings('ignore')

# 仅在Windows系统上设置中文字体
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows 微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

class BaseModelProcessor:
    def __init__(self, model_dir="output", config_dir="config"):
        self.model_dir = model_dir
        self.config_dir = config_dir
        self.feature_config = None
        self.category_features = None
        self.continuous_features = None
        self.train_feature_names = None
        self.lr_feature_names = None
        self.gbdt_model = None
        self.lr_model = None
        self.dl_model = None
        
    def load_feature_config(self):
        """加载特征配置"""
        try:
            # 加载特征配置文件
            features_path = os.path.join(self.config_dir, "features.csv")
            self.feature_config = pd.read_csv(features_path)
            print(f"✅ 特征配置已加载: {features_path}")
            
            # 分离类别特征和连续特征
            self.category_features = self.feature_config[
                self.feature_config['feature_type'] == 'category'
            ]['feature_name'].tolist()
            
            self.continuous_features = self.feature_config[
                self.feature_config['feature_type'] == 'continuous'
            ]['feature_name'].tolist()
            
            # 移除Id特征
            if 'Id' in self.continuous_features:
                self.continuous_features.remove('Id')
            
            print(f"📊 特征配置: {len(self.continuous_features)} 个连续特征, {len(self.category_features)} 个类别特征")
            return True
        except Exception as e:
            print(f"❌ 加载特征配置时出错: {e}")
            return False
    
    def load_models(self, model_type="gbdt_lr"):
        """加载训练好的模型"""
        try:
            if model_type == "gbdt_lr":
                # 加载GBDT模型
                gbdt_model_path = os.path.join(self.model_dir, "gbdt_model.pkl")
                self.gbdt_model = joblib.load(gbdt_model_path)
                print(f"✅ GBDT模型已加载: {gbdt_model_path}")
                
                # 加载LR模型
                lr_model_path = os.path.join(self.model_dir, "lr_model.pkl")
                self.lr_model = joblib.load(lr_model_path)
                print(f"✅ LR模型已加载: {lr_model_path}")
                
                # 获取LR模型使用的特征名称
                self.lr_feature_names = self.lr_model.feature_names_in_.tolist()
                print(f"✅ LR模型特征名称已加载: {len(self.lr_feature_names)} 个特征")
                
                # 加载训练时的特征名称
                train_features_path = os.path.join(self.model_dir, "train_feature_names.csv")
                train_features_df = pd.read_csv(train_features_path)
                self.train_feature_names = train_features_df['feature'].tolist()
                print(f"✅ 训练时特征名称已加载: {len(self.train_feature_names)} 个特征")
            elif model_type == "dl":
                if not HAS_TORCH:
                    print("❌ 未安装PyTorch，无法加载深度学习模型")
                    return False
                    
                # 加载深度学习模型信息
                model_info_path = os.path.join(self.model_dir, "dl_model_info.csv")
                if os.path.exists(model_info_path):
                    model_info = pd.read_csv(model_info_path).iloc[0]
                    self.dl_model_info = model_info.to_dict()
                    print(f"✅ 深度学习模型信息已加载: {model_info_path}")
                else:
                    print(f"⚠️ 深度学习模型信息文件不存在: {model_info_path}")
                    return False
                
                # 加载训练时的特征名称
                train_features_path = os.path.join(self.model_dir, "train_feature_names.csv")
                if os.path.exists(train_features_path):
                    train_features_df = pd.read_csv(train_features_path)
                    self.train_feature_names = train_features_df['feature'].tolist()
                    print(f"✅ 训练时特征名称已加载: {len(self.train_feature_names)} 个特征")
                else:
                    print(f"⚠️ 训练时特征名称文件不存在: {train_features_path}")
                    return False
            else:
                print(f"❌ 不支持的模型类型: {model_type}")
                return False
            
            return True
        except Exception as e:
            print(f"❌ 加载模型时出错: {e}")
            return False
    
    def save_models(self, gbdt_model=None, lr_model=None, category_feature=None, continuous_feature=None, 
                   dl_model=None, dl_model_info=None):
        """保存训练好的模型"""
        try:
            # 保存GBDT模型
            if gbdt_model is not None:
                joblib.dump(gbdt_model, os.path.join(self.model_dir, "gbdt_model.pkl"))
            
            # 保存LR模型
            if lr_model is not None:
                joblib.dump(lr_model, os.path.join(self.model_dir, "lr_model.pkl"))
            
            # 保存GBDT+LR模型相关信息
            if gbdt_model is not None:
                # 保存必要信息用于API服务
                pd.Series([gbdt_model.best_iteration_]).to_csv(
                    os.path.join(self.model_dir, "actual_n_estimators.csv"), 
                    index=False, 
                    header=['n_estimators']
                )
                
                pd.Series(gbdt_model.feature_name_).to_csv(
                    os.path.join(self.model_dir, "train_feature_names.csv"), 
                    index=False, 
                    header=['feature']
                )
                
                pd.Series(category_feature).to_csv(
                    os.path.join(self.model_dir, "category_features.csv"), 
                    index=False, 
                    header=['feature']
                )
                
                pd.Series(continuous_feature).to_csv(
                    os.path.join(self.model_dir, "continuous_features.csv"), 
                    index=False, 
                    header=['feature']
                )
            
            # 保存深度学习模型
            if dl_model is not None and HAS_TORCH:
                torch.save(dl_model.state_dict(), os.path.join(self.model_dir, "dl_model_best.pth"))
            
            # 保存深度学习模型信息
            if dl_model_info is not None:
                pd.DataFrame([dl_model_info]).to_csv(os.path.join(self.model_dir, "dl_model_info.csv"), index=False)
                print("✅ 深度学习模型信息已保存至 output/dl_model_info.csv")
            
            return True
        except Exception as e:
            print(f"❌ 保存模型时出错: {e}")
            return False
            
    def onehot_encode_features(self, df, columns, prefix_sep='_'):
        """对指定列进行One-Hot编码"""
        encoded_dfs = []
        for col in columns:
            if col in df.columns:
                # 对类别特征进行One-Hot编码
                onehot_df = pd.get_dummies(df[col], prefix=col, prefix_sep=prefix_sep)
                encoded_dfs.append(onehot_df)
        if encoded_dfs:
            return pd.concat(encoded_dfs, axis=1)
        return pd.DataFrame()
        
    def onehot_encode_leaf_features(self, leaf_indices, n_trees):
        """对叶子节点进行One-Hot编码"""
        leaf_dummies_list = []
        for i in range(n_trees):
            leaf_col_name = f"gbdt_leaf_{i}"
            leaf_series = pd.Series(leaf_indices[:, i], name=leaf_col_name)
            dummies = pd.get_dummies(leaf_series, prefix=leaf_col_name)
            leaf_dummies_list.append(dummies)
        leaf_dummies_combined = pd.concat(leaf_dummies_list, axis=1) if leaf_dummies_list else pd.DataFrame()
        return leaf_dummies_combined
        
    def match_features(self, df, required_features, fill_value=0):
        """确保数据框包含所需的特征"""
        # 获取当前特征
        current_features = set(df.columns)
        
        # 检查缺少的特征
        missing_features = set(required_features) - current_features
        if missing_features:
            print(f"⚠️  数据缺少 {len(missing_features)} 个特征")
            # 为缺少的特征添加默认值
            for feature in missing_features:
                df[feature] = fill_value
        
        # 移除多余的特征
        extra_features = current_features - set(required_features)
        if extra_features:
            print(f"ℹ️  移除 {len(extra_features)} 个多余的特征")
            df = df.drop(columns=list(extra_features))
        
        # 确保特征顺序与要求一致
        df = df[required_features]
        return df
        
    def show_model_interpretation_prompt(self):
        """
        显示提示信息，指导用户如何将模型训练日志复制到大模型进行解读
        """
        print("\n✅ ======================================")
        print("✅ 将下面的内容复制到大模型内进行解读（不包括此三行）")
        print("✅ ======================================\n")
        print("对以下(推荐/授信/预警)模型训练日志进行分析，输出银行业务人员可以理解的解读报告，目地是进行(推荐/授信/预警)，通过模型分析赋能业务决策。\n")
        
    def get_leaf_path_enhanced(self, booster, tree_index, leaf_index, feature_names, category_prefixes):
        """
        解析指定叶子节点的决策路径，支持翻译 one-hot 类别特征
        """
        try:
            model_dump = booster.dump_model()
            if tree_index >= len(model_dump['tree_info']):
                return None
            tree_info = model_dump['tree_info'][tree_index]['tree_structure']
        except Exception as e:
            print(f"获取树结构失败: {e}")
            return None

        node_stack = [(tree_info, [])]  # (当前节点, 路径列表)

        while node_stack:
            node, current_path = node_stack.pop()

            # 如果是目标叶子节点
            if 'leaf_index' in node and node['leaf_index'] == leaf_index:
                return current_path

            # 如果是分裂节点
            if 'split_feature' in node:
                feat_idx = node['split_feature']
                if feat_idx >= len(feature_names):
                    feat_name = f"Feature_{feat_idx}"
                else:
                    feat_name = feature_names[feat_idx]

                threshold = node.get('threshold', 0.0)
                decision_type = node.get('decision_type', '<=')

                # 检查是否为 one-hot 类别特征
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
                    # 类别特征通常用 > 0.5 判断是否激活
                    # 假设右子树是"等于该类别"
                    right_rule = f"{original_col} == '{category_value}'"
                    left_rule = f"{original_col} != '{category_value}'"
                else:
                    # 连续特征
                    if decision_type == '<=' or decision_type == 'no_greater':
                        right_rule = f"{feat_name} > {threshold:.4f}"
                        left_rule = f"{feat_name} <= {threshold:.4f}"
                    else:
                        right_rule = f"{feat_name} {decision_type} {threshold:.4f}"
                        left_rule = f"{feat_name} not {decision_type} {threshold:.4f}"

                # 添加左右子树到栈
                if 'right_child' in node:
                    node_stack.append((node['right_child'], current_path + [right_rule]))
                if 'left_child' in node:
                    node_stack.append((node['left_child'], current_path + [left_rule]))

        return None  # 未找到路径
        
    def calculate_ks_statistic(self, y_true, y_pred_prob):
        """
        计算 KS 统计量
        """
        # 将样本按预测概率排序
        data = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob})
        data_sorted = data.sort_values('y_pred_prob', ascending=False)
        
        # 计算累积分布
        cum_positive = (data_sorted['y_true'] == 1).cumsum() / (y_true == 1).sum()
        cum_negative = (data_sorted['y_true'] == 0).cumsum() / (y_true == 0).sum()
        
        # KS统计量是两个累积分布之间的最大差异
        ks_stat = np.max(np.abs(cum_positive - cum_negative))
        return ks_stat
        
    def plot_roc_curve(self, y_true, y_pred_prob, output_path="output/roc_curve.png"):
        """
        绘制 ROC 曲线
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        auc = roc_auc_score(y_true, y_pred_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ ROC曲线已保存至 {output_path}")
        return auc
        
    def analyze_feature_importance(self, booster, feature_names):
        """
        分析 GBDT 特征重要性（含影响方向）
        """
        # 获取 Gain 类型的重要性（更准确反映特征影响）
        gain_importance = booster.feature_importance(importance_type='gain')
        # 获取 Split 类型的重要性（特征被用于分裂的次数）
        split_importance = booster.feature_importance(importance_type='split')
        
        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Gain_Importance': gain_importance,
            'Split_Importance': split_importance
        }).sort_values('Gain_Importance', ascending=False)
        
        # 通过LightGBM内置功能分析特征影响方向
        try:
            # 这里需要训练数据来计算特征贡献值，所以在训练时调用
            # 此处仅返回基本的特征重要性信息
            feat_imp['Impact_Direction'] = 'Unknown'
            
        except Exception as e:
            # 如果分析失败，仍保留基本的特征重要性信息
            feat_imp['Impact_Direction'] = 'Unknown'
            
        return feat_imp

    def plot_training_curves(self, train_losses, val_losses, train_aucs, val_aucs, output_path="output/training_curves.png"):
        """
        绘制训练曲线（损失和AUC）
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_aucs, label='Train AUC')
        plt.plot(val_aucs, label='Validation AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Training and Validation AUC')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 训练曲线已保存至 {output_path}")

    def analyze_dl_feature_importance(self, model, x_val_tensor, x_train_columns, output_path="output/dl_feature_importance.csv"):
        """
        分析深度学习模型的特征重要性
        """
        # 完全移除深度学习相关功能，避免在没有PyTorch时出现错误
        print("❌ 未安装PyTorch，已禁用深度学习模型相关功能")
        return None
