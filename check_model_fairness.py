import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from base.base_fairness_processor import BaseFairnessProcessor
from base.base_model_processor import BaseModelProcessor

def calculate_fairness_metrics():
    """
    计算模型的公平性指标
    """
    print("⚖️  开始计算模型公平性指标...")
    
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # ========== 1. 加载训练好的模型和配置 ==========
    try:
        # 初始化基础模型处理器
        processor = BaseModelProcessor()
        
        # 加载特征配置
        if not processor.load_feature_config():
            print("❌ 特征配置加载失败")
            return
        
        # 加载GBDT模型和LR模型
        if not processor.load_models():
            print("❌ 模型加载失败")
            return
            
        gbdt_model = processor.gbdt_model
        lr_model = processor.lr_model
        category_features = processor.category_features
        continuous_features = processor.continuous_features
        train_feature_names = processor.train_feature_names
        
        print(f"✅ 模型和配置加载成功")
        print(f"   - 类别特征: {len(category_features)} 个")
        print(f"   - 连续特征: {len(continuous_features)} 个")
        print(f"   - 训练特征: {len(train_feature_names)} 个")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # ========== 2. 加载验证数据 ==========
    try:
        # 加载训练数据
        data = pd.read_csv('data_train/data.csv')
        print(f"✅ 训练数据加载成功: {data.shape}")
        
        # 分离特征和标签
        target = data.pop('Label')
        train_data = data.copy()
        
        # 确保训练数据都是数值类型
        for col in train_data.columns:
            if train_data[col].dtype == 'bool':
                train_data[col] = train_data[col].astype(int)
            elif train_data[col].dtype == 'object':
                # 尝试转换对象类型的列
                train_data[col] = pd.to_numeric(train_data[col], errors='coerce').fillna(-1)
        
        # 保存一份原始数据用于敏感属性分析
        original_train_data = train_data.copy()
        
        # 对类别特征进行One-Hot编码（与训练时保持一致）
        print("正在进行类别特征One-Hot编码...")
        encoded_features = []
        remaining_features = train_data.columns.tolist()
        
        # 处理类别特征
        for col in category_features:
            if col in train_data.columns:
                # 对类别特征进行One-Hot编码
                onehot_df = pd.get_dummies(train_data[col], prefix=col)
                encoded_features.append(onehot_df)
                remaining_features.remove(col)
        
        # 合并编码后的特征和剩余特征
        if encoded_features:
            encoded_data = pd.concat(encoded_features, axis=1)
            remaining_data = train_data[remaining_features]
            train_data = pd.concat([remaining_data, encoded_data], axis=1)
            print(f"✅ One-Hot编码完成: {len(encoded_features)} 个类别特征被编码")
        else:
            print("ℹ️  未发现需要编码的类别特征")
        
        # 确保特征顺序与训练时一致
        # 获取训练时的特征名称
        train_feature_set = set(train_feature_names)
        current_feature_set = set(train_data.columns)
        
        # 检查缺失的特征
        missing_features = train_feature_set - current_feature_set
        if missing_features:
            print(f"⚠️  发现 {len(missing_features)} 个缺失特征，将用0填充")
            for feature in missing_features:
                train_data[feature] = 0
        
        # 移除多余的特征
        extra_features = current_feature_set - train_feature_set
        if extra_features:
            print(f"⚠️  移除 {len(extra_features)} 个多余特征")
            train_data = train_data.drop(columns=list(extra_features))
        
        # 按训练时的特征顺序重新排列
        train_data = train_data[train_feature_names]
        print(f"✅ 特征对齐完成: {train_data.shape[1]} 个特征")
        
        # 划分训练/验证集（使用与训练时相同的随机种子）
        x_train, x_val, y_train, y_val = train_test_split(
            train_data, target, test_size=0.2, random_state=2020, stratify=target
        )
        print(f"✅ 数据集划分完成: 训练集 {x_train.shape}, 验证集 {x_val.shape}")
        
        # 同时对原始数据进行相同的划分，用于敏感属性分析
        _, original_x_val, _, _ = train_test_split(
            original_train_data, target, test_size=0.2, random_state=2020, stratify=target
        )
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 3. 获取叶子节点特征 ==========
    try:
        # 获取GBDT模型实际训练的树数量
        actual_n_estimators = gbdt_model.best_iteration_
        print(f"✅ GBDT实际树数量: {actual_n_estimators}")
        
        # 获取叶子节点索引
        print("正在获取叶子节点索引...")
        gbdt_feats_train = gbdt_model.booster_.predict(x_train.values, pred_leaf=True)
        gbdt_feats_val = gbdt_model.booster_.predict(x_val.values, pred_leaf=True)
        
        # 对叶子节点做 One-Hot 编码
        gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(actual_n_estimators)]
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
        
        # 划分LR训练/验证集（使用与训练时相同的随机种子）
        x_train_lr, x_val_lr, y_train_lr, y_val_lr = train_test_split(
            train_lr, y_train, test_size=0.3, random_state=2018, stratify=y_train
        )
        print("✅ 叶子节点特征处理完成")
        
    except Exception as e:
        print(f"❌ 叶子节点特征处理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 4. 计算公平性指标 ==========
    print("\n" + "="*60)
    print("⚖️  正在计算模型公平性指标...")
    print("="*60)
    
    # 选择一个类别特征作为敏感属性
    sensitive_attr = None
    
    # 首先尝试使用类别特征作为敏感属性
    if category_features:
        # 选择一个具有更多样化值的类别特征作为敏感属性
        # 优先选择部门相关的特征，因为它们可能具有更多不同的值
        department_features = [col for col in category_features if '部门' in col]
        if department_features:
            sensitive_col = department_features[0]  # 选择第一个部门特征
        else:
            sensitive_col = category_features[0]  # 否则选择第一个类别特征
        
        print(f"ℹ️  使用 '{sensitive_col}' 作为敏感属性进行公平性分析")
        
        try:
            # 从已保存的原始数据中获取敏感属性列的值
            if sensitive_col in original_x_val.columns:
                # 获取敏感属性值
                sensitive_attr = original_x_val[sensitive_col]
                unique_values = sensitive_attr.unique()
                print(f"✅ 敏感属性列 '{sensitive_col}' 已加载，包含 {len(unique_values)} 个唯一值: {unique_values}")
                
                # 检查是否有足够的多样性来进行公平性分析
                if len(unique_values) < 2:
                    print("⚠️  敏感属性列的唯一值过少，无法进行有效的公平性分析")
                    sensitive_attr = None
            else:
                print(f"⚠️  敏感属性列 '{sensitive_col}' 在原始数据中未找到")
        except Exception as e:
            print(f"⚠️  加载敏感属性时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 如果没有找到合适的类别特征，创建一个人工的敏感属性列用于演示
    if sensitive_attr is None:
        print("ℹ️  未找到合适的类别特征作为敏感属性，创建人工敏感属性列用于演示")
        try:
            # 使用LR模型的预测结果创建一个人工的敏感属性
            # 基于预测概率的中位数将样本分为两组
            # 为了避免特征名称不匹配的问题，我们先检查x_val_lr的特征名称
            print(f"ℹ️  x_val_lr 特征数量: {x_val_lr.shape[1]}")
            print(f"ℹ️  LR模型期望的特征数量: {len(lr_model.feature_names_in_)}")
            
            # 确保特征名称匹配
            if set(x_val_lr.columns) == set(lr_model.feature_names_in_):
                # 重新排列列以匹配LR模型期望的顺序
                x_val_lr_aligned = x_val_lr[lr_model.feature_names_in_]
                y_val_pred_prob_temp = lr_model.predict_proba(x_val_lr_aligned)[:, 1]
                median_prob = np.median(y_val_pred_prob_temp)
                sensitive_attr = pd.Series([1 if prob >= median_prob else 0 for prob in y_val_pred_prob_temp])
                print(f"✅ 人工敏感属性列已创建，包含 {len(sensitive_attr.unique())} 个唯一值: {sensitive_attr.unique()}")
            else:
                print("⚠️  特征名称不匹配，无法使用LR模型预测结果创建敏感属性")
                # 最后的备选方案：基于样本索引创建
                sensitive_attr = pd.Series([i % 2 for i in range(len(x_val_lr))])
                print(f"✅ 人工敏感属性列已创建，包含 {len(sensitive_attr.unique())} 个唯一值: {sensitive_attr.unique()}")
        except Exception as e:
            print(f"⚠️  创建人工敏感属性时出错: {e}")
            # 最后的备选方案：基于样本索引创建
            try:
                sensitive_attr = pd.Series([i % 2 for i in range(len(x_val_lr))])
                print(f"✅ 人工敏感属性列已创建，包含 {len(sensitive_attr.unique())} 个唯一值: {sensitive_attr.unique()}")
            except Exception as e2:
                print(f"⚠️  创建人工敏感属性时出错: {e2}")
    
    if sensitive_attr is not None and len(sensitive_attr) == len(y_val_lr):
        try:
            # 确保特征名称匹配后再进行预测
            if set(x_val_lr.columns) == set(lr_model.feature_names_in_):
                # 重新排列列以匹配LR模型期望的顺序
                x_val_lr_aligned = x_val_lr[lr_model.feature_names_in_]
                # 使用LR模型进行预测
                y_val_pred_prob = lr_model.predict_proba(x_val_lr_aligned)[:, 1]
            else:
                print("⚠️  特征名称不匹配，无法进行预测")
                y_val_pred_prob = None
                
            if y_val_pred_prob is not None:
                # 计算各种公平性指标
                fairness_metrics = BaseFairnessProcessor.calculate_fairness_metrics(
                    y_val_lr.values, 
                    y_val_pred_prob, 
                    sensitive_attr.values
                )
                
                if fairness_metrics is not None:
                    # 输出公平性指标结果
                    print("\n📊 模型公平性指标:")
                    for _, row in fairness_metrics.iterrows():
                        print(f"   {row['Metric']}: {row['Score']:.4f}")
                    
                    # 保存公平性指标到CSV文件
                    fairness_metrics.to_csv('output/fairness_metrics.csv', index=False)
                    print("\n✅ 公平性指标已保存至 output/fairness_metrics.csv")
                    
                    # 计算并显示AUC作为参考
                    val_auc = roc_auc_score(y_val_lr, y_val_pred_prob)
                    print(f"✅ 验证集 AUC: {val_auc:.4f}")
                else:
                    print("⚠️  公平性指标计算失败")
            else:
                print("⚠️  无法进行预测，跳过公平性指标计算")
        except Exception as e:
            print(f"⚠️  计算公平性指标时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  未找到有效的敏感属性列，无法计算公平性指标")
        print("ℹ️  为了进行公平性分析，请确保数据中包含合适的敏感属性列（如性别、年龄组等）")

if __name__ == "__main__":
    calculate_fairness_metrics()