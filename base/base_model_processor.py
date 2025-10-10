import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

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
    
    def load_models(self):
        """加载训练好的模型"""
        try:
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
            
            return True
        except Exception as e:
            print(f"❌ 加载模型时出错: {e}")
            return False
    
    def save_models(self, gbdt_model, lr_model, category_feature, continuous_feature):
        """保存训练好的模型"""
        try:
            # 保存GBDT模型
            joblib.dump(gbdt_model, os.path.join(self.model_dir, "gbdt_model.pkl"))
            # 保存LR模型
            joblib.dump(lr_model, os.path.join(self.model_dir, "lr_model.pkl"))
            
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
