import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ========== 基础公平性处理器 ==========
class BaseFairnessProcessor:
    """公平性检测处理器"""
    
    @staticmethod
    def calculate_demographic_parity(y_pred, sensitive_attr):
        """
        计算 Demographic Parity (人口统计学公平性)
        """
        groups = np.unique(sensitive_attr)
        positive_rates = []
        
        for group in groups:
            mask = sensitive_attr == group
            positive_rate = np.mean(y_pred[mask])
            positive_rates.append(positive_rate)
        
        # 计算最大差异作为不公平度量
        dp_diff = np.max(positive_rates) - np.min(positive_rates)
        return 1 - dp_diff  # 转换为公平性得分 (0-1之间，1表示完全公平)

    @staticmethod
    def calculate_equal_opportunity(y_true, y_pred, sensitive_attr):
        """
        计算 Equal Opportunity (机会均等)
        """
        groups = np.unique(sensitive_attr)
        tpr_rates = []
        
        for group in groups:
            mask = (sensitive_attr == group) & (y_true == 1)
            if np.sum(mask) > 0:
                tpr = np.mean(y_pred[mask])
            else:
                tpr = 0
            tpr_rates.append(tpr)
        
        # 计算最大差异作为不公平度量
        eo_diff = np.max(tpr_rates) - np.min(tpr_rates)
        return 1 - eo_diff  # 转换为公平性得分 (0-1之间，1表示完全公平)

    @staticmethod
    def calculate_equalized_odds(y_true, y_pred, sensitive_attr):
        """
        计算 Equalized Odds (均衡几率)
        """
        groups = np.unique(sensitive_attr)
        tpr_rates = []
        fpr_rates = []
        
        for group in groups:
            # 计算真阳性率 (TPR)
            tp_mask = (sensitive_attr == group) & (y_true == 1)
            if np.sum(tp_mask) > 0:
                tpr = np.mean(y_pred[tp_mask])
            else:
                tpr = 0
            tpr_rates.append(tpr)
            
            # 计算假阳性率 (FPR)
            fp_mask = (sensitive_attr == group) & (y_true == 0)
            if np.sum(fp_mask) > 0:
                fpr = np.mean(y_pred[fp_mask])
            else:
                fpr = 0
            fpr_rates.append(fpr)
        
        # 计算TPR和FPR的最大差异
        tpr_diff = np.max(tpr_rates) - np.min(tpr_rates)
        fpr_diff = np.max(fpr_rates) - np.min(fpr_rates)
        
        # 综合不公平度量
        eo_diff = (tpr_diff + fpr_diff) / 2
        return 1 - eo_diff  # 转换为公平性得分 (0-1之间，1表示完全公平)

    @staticmethod
    def calculate_predictive_parity(y_true, y_pred, sensitive_attr):
        """
        计算 Predictive Parity (预测公平性)
        """
        groups = np.unique(sensitive_attr)
        ppv_rates = []
        
        for group in groups:
            # 计算预测值为正的样本中真实值为正的比例 (Positive Predictive Value, PPV)
            pred_pos_mask = y_pred == 1
            group_mask = sensitive_attr == group
            combined_mask = pred_pos_mask & group_mask
            
            if np.sum(combined_mask) > 0:
                ppv = np.mean(y_true[combined_mask])
            else:
                ppv = 0
            ppv_rates.append(ppv)
        
        # 计算最大差异作为不公平度量
        pp_diff = np.max(ppv_rates) - np.min(ppv_rates)
        return 1 - pp_diff  # 转换为公平性得分 (0-1之间，1表示完全公平)

    @staticmethod
    def calculate_fairness_metrics(y_true, y_pred, sensitive_attr):
        """
        计算所有公平性指标
        """
        if sensitive_attr is None:
            return None
            
        # 将预测概率转换为二值预测（阈值设为0.5）
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        # 计算各种公平性指标
        demographic_parity = BaseFairnessProcessor.calculate_demographic_parity(y_pred_binary, sensitive_attr)
        equal_opportunity = BaseFairnessProcessor.calculate_equal_opportunity(y_true, y_pred_binary, sensitive_attr)
        equalized_odds = BaseFairnessProcessor.calculate_equalized_odds(y_true, y_pred_binary, sensitive_attr)
        predictive_parity = BaseFairnessProcessor.calculate_predictive_parity(y_true, y_pred_binary, sensitive_attr)
        
        # 返回结果
        fairness_metrics = pd.DataFrame({
            'Metric': ['Demographic Parity', 'Equal Opportunity', 'Equalized Odds', 'Predictive Parity'],
            'Score': [demographic_parity, equal_opportunity, equalized_odds, predictive_parity]
        })
        
        return fairness_metrics

# ========== 数据加载器 ==========
class DataLoader:
    def __init__(self):
        self.data = None
        self.target = None
        self.train_data = None
        self.x_train = None
        self.x_val = None
        self.y_train = None
        self.y_val = None
        
    def load_training_data(self, data_path='data_train/data.csv'):
        """加载训练数据"""
        try:
            # 加载训练数据
            self.data = pd.read_csv(data_path)
            print(f"✅ 训练数据加载成功: {self.data.shape}")
            
            # 分离特征和标签
            self.target = self.data.pop('Label')
            self.train_data = self.data.copy()
            
            # 确保训练数据都是数值类型
            for col in self.train_data.columns:
                if self.train_data[col].dtype == 'bool':
                    self.train_data[col] = self.train_data[col].astype(int)
                elif self.train_data[col].dtype == 'object':
                    # 尝试转换对象类型的列
                    self.train_data[col] = pd.to_numeric(self.train_data[col], errors='coerce').fillna(-1)
            
            return True
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def split_data(self, test_size=0.2, random_state=2020):
        """划分训练/验证集"""
        try:
            # 划分训练/验证集，保持索引
            x_train, x_val, y_train, y_val = train_test_split(
                self.train_data, self.target, test_size=test_size, random_state=random_state, stratify=self.target
            )
            
            # 重置索引，但保留原索引作为列
            self.x_train = x_train.reset_index(drop=False)
            self.x_val = x_val.reset_index(drop=False)
            self.y_train = y_train.reset_index(drop=False)
            self.y_val = y_val.reset_index(drop=False)
            
            print(f"✅ 数据集划分完成: 训练集 {self.x_train.shape}, 验证集 {self.x_val.shape}")
            return True
        except Exception as e:
            print(f"❌ 数据集划分失败: {e}")
            return False

# ========== 特征处理器 ==========
class FeatureProcessor:
    def __init__(self, model_dir="output", config_dir="config"):
        self.model_dir = model_dir
        self.config_dir = config_dir
        self.category_features = []
        self.continuous_features = []
        self.train_feature_names = []
        
    def load_feature_config(self):
        """加载特征配置"""
        try:
            # 加载特征配置文件
            features_path = os.path.join(self.config_dir, "features.csv")
            features_df = pd.read_csv(features_path)
            self.category_features = features_df[features_df['feature_type'] == 'category']['feature_name'].tolist()
            self.continuous_features = features_df[features_df['feature_type'] == 'continuous']['feature_name'].tolist()
            print(f"✅ 特征配置加载成功")
            print(f"   - 类别特征: {len(self.category_features)} 个")
            print(f"   - 连续特征: {len(self.continuous_features)} 个")
            
            # 加载训练时的特征名称
            train_features_path = os.path.join(self.model_dir, "train_feature_names.csv")
            train_features_df = pd.read_csv(train_features_path)
            self.train_feature_names = train_features_df['feature'].tolist()
            print(f"✅ 训练时特征名称已加载: {len(self.train_feature_names)} 个特征")
            return True
        except Exception as e:
            print(f"❌ 特征配置加载失败: {e}")
            return False
    
    def encode_categorical_features(self, data):
        """对类别特征进行One-Hot编码"""
        try:
            print("正在进行类别特征One-Hot编码...")
            encoded_features = []
            remaining_features = data.columns.tolist()
            
            # 处理类别特征
            for col in self.category_features:
                if col in data.columns:
                    # 对类别特征进行One-Hot编码
                    onehot_df = pd.get_dummies(data[col], prefix=col)
                    encoded_features.append(onehot_df)
                    remaining_features.remove(col)
            
            # 合并编码后的特征和剩余特征
            if encoded_features:
                encoded_data = pd.concat(encoded_features, axis=1)
                remaining_data = data[remaining_features]
                processed_data = pd.concat([remaining_data, encoded_data], axis=1)
                print(f"✅ One-Hot编码完成: {len(encoded_features)} 个类别特征被编码")
                return processed_data
            else:
                print("ℹ️  未发现需要编码的类别特征")
                return data
        except Exception as e:
            print(f"❌ 类别特征编码失败: {e}")
            return None
    
    def align_features(self, data):
        """确保特征顺序与训练时一致"""
        try:
            # 获取训练时的特征名称
            train_feature_set = set(self.train_feature_names)
            current_feature_set = set(data.columns)
            
            # 检查缺失的特征
            missing_features = train_feature_set - current_feature_set
            if missing_features:
                print(f"⚠️  发现 {len(missing_features)} 个缺失特征，将用0填充")
                for feature in missing_features:
                    data[feature] = 0
            
            # 移除多余的特征
            extra_features = current_feature_set - train_feature_set
            if extra_features:
                print(f"⚠️  移除 {len(extra_features)} 个多余特征")
                data = data.drop(columns=list(extra_features))
            
            # 按训练时的特征顺序重新排列
            data = data[self.train_feature_names]
            print(f"✅ 特征对齐完成: {data.shape[1]} 个特征")
            return data
        except Exception as e:
            print(f"❌ 特征对齐失败: {e}")
            return None

# ========== 模型处理器 ==========
class ModelHandler:
    def __init__(self):
        self.gbdt_model = None
        self.lr_model = None
        self.actual_n_estimators = 0
        self.lr_feature_names = []
        
    def load_models(self, gbdt_path='output/gbdt_model.pkl', lr_path='output/lr_model.pkl'):
        """加载GBDT和LR模型"""
        try:
            self.gbdt_model = joblib.load(gbdt_path)
            self.lr_model = joblib.load(lr_path)
            
            # 获取LR模型使用的特征名称
            self.lr_feature_names = self.lr_model.feature_names_in_.tolist()
            print(f"✅ 模型加载成功")
            print(f"✅ LR模型特征名称已加载: {len(self.lr_feature_names)} 个特征")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def get_leaf_features(self, x_train, x_val):
        """获取叶子节点特征"""
        try:
            # 获取GBDT模型实际训练的树数量
            self.actual_n_estimators = self.gbdt_model.best_iteration_
            print(f"✅ GBDT实际树数量: {self.actual_n_estimators}")
            
            # 获取叶子节点索引
            print("正在获取叶子节点索引...")
            gbdt_feats_train = self.gbdt_model.booster_.predict(x_train.values, pred_leaf=True)
            gbdt_feats_val = self.gbdt_model.booster_.predict(x_val.values, pred_leaf=True)
            
            # 对叶子节点做 One-Hot 编码
            gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(self.actual_n_estimators)]
            df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
            df_val_gbdt_feats = pd.DataFrame(gbdt_feats_val, columns=gbdt_feats_name)
            
            data_gbdt = pd.concat([df_train_gbdt_feats, df_val_gbdt_feats], ignore_index=True)
            
            for col in gbdt_feats_name:
                # 确保列数据是整数类型
                data_gbdt[col] = data_gbdt[col].astype(int)
                onehot_feats = pd.get_dummies(data_gbdt[col], prefix=col)
                data_gbdt.drop([col], axis=1, inplace=True)
                data_gbdt = pd.concat([data_gbdt, onehot_feats], axis=1)
            
            # 确保所有列都是数值类型
            for col in data_gbdt.columns:
                if data_gbdt[col].dtype == 'bool':
                    data_gbdt[col] = data_gbdt[col].astype(int)
                # 检查是否有字符串类型的列
                elif data_gbdt[col].dtype == 'object':
                    print(f"⚠️  发现非数值列: {col}, 类型: {data_gbdt[col].dtype}")
                    # 尝试转换为数值类型
                    data_gbdt[col] = pd.to_numeric(data_gbdt[col], errors='coerce').fillna(0)
            
            train_len = df_train_gbdt_feats.shape[0]
            train_lr = data_gbdt.iloc[:train_len, :].reset_index(drop=True)
            val_lr = data_gbdt.iloc[train_len:, :].reset_index(drop=True)
            
            # 划分LR训练/验证集，返回正确的索引
            x_train_lr, x_val_lr, y_train_lr_indices, y_val_lr_indices = train_test_split(
                train_lr, range(len(train_lr)), test_size=0.3, random_state=2018
            )
            
            # 转换索引为numpy数组
            y_train_lr = np.array(y_train_lr_indices)
            y_val_lr = np.array(y_val_lr_indices)
            
            print("✅ 叶子节点特征处理完成")
            
            return x_train_lr, x_val_lr, y_train_lr, y_val_lr
        except Exception as e:
            print(f"❌ 叶子节点特征处理失败: {e}")
            return None, None, None, None
    
    def align_lr_features(self, data):
        """确保数据特征与LR模型匹配"""
        try:
            # 获取当前特征
            current_features = set(data.columns)
            
            # 检查缺少的特征
            missing_features = set(self.lr_feature_names) - current_features
            if missing_features:
                print(f"⚠️  数据缺少 {len(missing_features)} 个特征")
                # 为缺少的特征添加默认值0
                for feature in missing_features:
                    data[feature] = 0
            
            # 移除多余的特征
            extra_features = current_features - set(self.lr_feature_names)
            if extra_features:
                print(f"ℹ️  移除 {len(extra_features)} 个多余的特征")
                data = data.drop(columns=list(extra_features))
            
            # 确保特征顺序与LR模型一致
            data = data[self.lr_feature_names]
            return data
        except Exception as e:
            print(f"❌ LR特征对齐失败: {e}")
            return None

# ========== 公平性计算器 ==========
class FairnessCalculator:
    def __init__(self):
        self.sensitive_attr = None
        
    def load_sensitive_config(self, config_path='config/sensitive_attr.csv'):
        """从配置文件加载敏感属性配置"""
        try:
            # 读取敏感属性配置文件
            config_df = pd.read_csv(config_path)
            if len(config_df) > 0:
                config = config_df.iloc[0]  # 取第一行配置
                return config['file_name'], config['sheet_name'], config['column_name']
            else:
                print("❌ 敏感属性配置文件为空")
                return None, None, None
        except Exception as e:
            print(f"❌ 敏感属性配置加载失败: {e}")
            return None, None, None
    
    def load_sensitive_attribute(self, file_name, column_name, sheet_name=None):
        """从指定文件加载敏感属性"""
        try:
            # 读取敏感属性文件
            if sheet_name and pd.notna(sheet_name):
                sensitive_data = pd.read_excel(f'data_train/{file_name}', sheet_name=sheet_name)
            else:
                sensitive_data = pd.read_excel(f'data_train/{file_name}')
            print(f"✅ 敏感属性文件加载成功: {sensitive_data.shape}")
            
            # 获取敏感属性列
            if column_name in sensitive_data.columns:
                self.sensitive_attr = sensitive_data[column_name]
                unique_values = self.sensitive_attr.unique()
                print(f"✅ 敏感属性列 '{column_name}' 已加载，包含 {len(unique_values)} 个唯一值: {unique_values}")
                
                # 检查是否有足够的多样性来进行公平性分析
                if len(unique_values) < 2:
                    print("⚠️  敏感属性列的唯一值过少，无法进行有效的公平性分析")
                    return False
                return True
            else:
                print(f"❌ 敏感属性列 '{column_name}' 在文件中未找到")
                return False
        except Exception as e:
            print(f"❌ 敏感属性加载失败: {e}")
            return False
    
    def calculate_fairness(self, y_true, y_pred_prob, sensitive_attr):
        """计算公平性指标"""
        try:
            # 将预测概率转换为二值预测（阈值设为0.5）
            y_pred_binary = (y_pred_prob >= 0.5).astype(int)
            
            # 计算各种公平性指标
            fairness_metrics = BaseFairnessProcessor.calculate_fairness_metrics(
                y_true, y_pred_prob, sensitive_attr
            )
            
            if fairness_metrics is not None:
                # 输出公平性指标结果
                print("\n📊 模型公平性指标:")
                for _, row in fairness_metrics.iterrows():
                    print(f"   {row['Metric']}: {row['Score']:.4f}")
                
                # 保存公平性指标到CSV文件
                fairness_metrics.to_csv('output/fairness_metrics.csv', index=False)
                print("\n✅ 公平性指标已保存至 output/fairness_metrics.csv")
                
                return True
            else:
                print("⚠️  公平性指标计算失败")
                return False
        except Exception as e:
            print(f"❌ 公平性指标计算失败: {e}")
            return False

# ========== 主函数 ==========
def calculate_fairness_metrics():
    """
    计算模型的公平性指标
    """
    print("⚖️  开始计算模型公平性指标...")
    
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # ========== 1. 初始化各模块 ==========
    data_loader = DataLoader()
    feature_processor = FeatureProcessor()
    model_handler = ModelHandler()
    fairness_calculator = FairnessCalculator()
    
    # ========== 2. 加载模型和配置 ==========
    try:
        # 加载特征配置
        if not feature_processor.load_feature_config():
            print("❌ 特征配置加载失败")
            return
        
        # 加载GBDT模型和LR模型
        if not model_handler.load_models():
            print("❌ 模型加载失败")
            return
            
        print(f"✅ 模型和配置加载成功")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # ========== 3. 加载和处理数据 ==========
    try:
        # 加载训练数据
        if not data_loader.load_training_data():
            print("❌ 训练数据加载失败")
            return
        
        # 划分训练/验证集
        if not data_loader.split_data():
            print("❌ 数据集划分失败")
            return
        
        # 保存原始索引
        original_x_train_indices = data_loader.x_train['index'].values
        original_x_val_indices = data_loader.x_val['index'].values
        original_y_train_indices = data_loader.y_train['index'].values
        original_y_val_indices = data_loader.y_val['index'].values
        
        # 对类别特征进行One-Hot编码
        encoded_data = feature_processor.encode_categorical_features(data_loader.train_data)
        if encoded_data is None:
            print("❌ 类别特征编码失败")
            return
        
        # 确保特征顺序与训练时一致
        aligned_data = feature_processor.align_features(encoded_data)
        if aligned_data is None:
            print("❌ 特征对齐失败")
            return
        
        # 更新数据加载器中的数据，使用原始索引
        data_loader.x_train = aligned_data.iloc[original_x_train_indices].reset_index(drop=True)
        data_loader.x_val = aligned_data.iloc[original_x_val_indices].reset_index(drop=True)
        data_loader.y_train = data_loader.target.iloc[original_y_train_indices].reset_index(drop=True)
        data_loader.y_val = data_loader.target.iloc[original_y_val_indices].reset_index(drop=True)
        
    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        return
    
    # ========== 4. 获取叶子节点特征 ==========
    try:
        x_train_lr, x_val_lr, y_train_lr, y_val_lr = model_handler.get_leaf_features(
            data_loader.x_train, data_loader.x_val
        )
        if x_train_lr is None:
            print("❌ 叶子节点特征处理失败")
            return
        
        # 确保叶子节点特征与LR模型匹配
        x_val_lr_aligned = model_handler.align_lr_features(x_val_lr)
        if x_val_lr_aligned is None:
            print("❌ LR特征对齐失败")
            return
        
    except Exception as e:
        print(f"❌ 叶子节点特征处理失败: {e}")
        return
    
    # ========== 5. 加载敏感属性 ==========
    # 从配置文件读取敏感属性配置
    sensitive_file, sheet_name, sensitive_column = fairness_calculator.load_sensitive_config()
    
    if not sensitive_file or not sensitive_column:
        print("❌ 无法从配置文件加载敏感属性配置")
        return
    
    print(f"ℹ️  使用配置文件中的敏感属性: 文件='{sensitive_file}', 表='{sheet_name}', 列='{sensitive_column}'")
    
    if not fairness_calculator.load_sensitive_attribute(sensitive_file, sensitive_column, sheet_name):
        print("❌ 敏感属性加载失败")
        return
    
    # ========== 6. 计算公平性指标 ==========
    print("\n" + "="*60)
    print("⚖️  正在计算模型公平性指标...")
    print("="*60)
    
    try:
        # 使用LR模型进行预测
        y_val_pred_prob = model_handler.lr_model.predict_proba(x_val_lr_aligned)[:, 1]
        print("✅ 预测完成")
        
        # 计算公平性指标
        # 修复敏感属性索引匹配问题
        # 获取原始验证集的标签值
        original_y_val = data_loader.target.iloc[original_y_val_indices].values
        # 确保y_val_lr的索引在有效范围内
        valid_indices_mask = y_val_lr < len(original_y_val)
        valid_indices = y_val_lr[valid_indices_mask]
        y_val_for_fairness = original_y_val[valid_indices]
        y_val_pred_prob_filtered = y_val_pred_prob[valid_indices_mask]
        
        # 获取对应的敏感属性值
        sensitive_attr_values = fairness_calculator.sensitive_attr.values[valid_indices]
        
        fairness_calculator.calculate_fairness(
            y_val_for_fairness, 
            y_val_pred_prob_filtered, 
            sensitive_attr_values
        )
        
    except Exception as e:
        print(f"❌ 公平性指标计算失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    calculate_fairness_metrics()