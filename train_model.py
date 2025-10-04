import os
os.environ["NUMBA_DISABLE_TBB"] = "1"
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import log_evaluation
import matplotlib.pyplot as plt
import joblib
import platform

# 仅在Windows系统上设置中文字体
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows 微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ========== 工具函数：解析叶子节点路径（增强版） ==========
def get_leaf_path_enhanced(booster, tree_index, leaf_index, feature_names, category_prefixes):
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
                # 假设右子树是“等于该类别”
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


# ========== 数据预处理 ==========
def preProcess():
    path = 'data_train/'
    try:
        df_train = pd.read_csv(path + 'train.csv', encoding='utf-8')
    except UnicodeDecodeError:
        print("⚠️ UTF-8 解码失败，尝试使用 GBK 编码...")
        df_train = pd.read_csv(path + 'train.csv', encoding='gbk')
    
    df_train.drop(['Id'], axis=1, inplace=True)
    data = df_train.fillna(-1)
    
    data.to_csv('data_train/data.csv', index=False, encoding='utf-8')
    return data


# ========== GBDT + LR 核心训练函数 ==========
def gbdt_lr_train(data, category_feature, continuous_feature):
    """
    使用 GBDT + LR 训练模型，增强可解释性输出
    """
    # 创建输出目录
    os.makedirs('output', exist_ok=True)

    # ========== Step 1: 类别特征 One-Hot 编码 ==========
    for col in category_feature:
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category')
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    # 分离特征和标签
    target = data.pop('Label')
    train = data.copy()

    # 划分训练/验证集
    x_train, x_val, y_train, y_val = train_test_split(
        train, target, test_size=0.2, random_state=2020, stratify=target
    )

    # ========== Step 2: 训练 GBDT ==========
    n_estimators = 32
    num_leaves = 64

    model = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        subsample=0.8,
        min_child_weight=0.1,
        min_child_samples=10,
        colsample_bytree=0.7,
        num_leaves=num_leaves,
        learning_rate=0.05,
        n_estimators=n_estimators,
        random_state=2020,
        n_jobs=-1
    )

    model.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[
            log_evaluation(0),
            lgb.early_stopping(stopping_rounds=5, verbose=False)
        ]
    )

    # ========== 🆕 获取实际训练的树数量 ==========
    actual_n_estimators = model.best_iteration_
    print(f"✅ 实际训练树数量: {actual_n_estimators} (原计划: {n_estimators})")

    # ========== Step 2.5: 输出 GBDT 特征重要性 ==========
    # 获取 Gain 类型的重要性（更准确反映特征影响）
    gain_importance = model.booster_.feature_importance(importance_type='gain')
    # 获取 Split 类型的重要性（特征被用于分裂的次数）
    split_importance = model.booster_.feature_importance(importance_type='split')
    
    feat_imp = pd.DataFrame({
        'Feature': x_train.columns,
        'Gain_Importance': gain_importance,
        'Split_Importance': split_importance
    }).sort_values('Gain_Importance', ascending=False)

    print("\n" + "="*60)
    print("📊 GBDT Top 20 重要特征 (按 Gain 排序):")
    print("="*60)
    print(feat_imp.head(20))
    feat_imp.to_csv('output/gbdt_feature_importance.csv', index=False)
    print("✅ 已保存至 output/gbdt_feature_importance.csv")
    
    # ========== 增加：通过SHAP值分析特征影响方向 ==========
    try:        
        import shap
        
        print("\n" + "="*60)
        print("🧠 正在通过SHAP分析特征影响方向...")
        print("="*60)
        
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model.booster_)
        
        # 为了提高效率，只使用一部分数据计算SHAP值
        sample_size = min(100, len(x_train))
        x_train_sample = x_train.iloc[:sample_size]
        shap_values = explainer.shap_values(x_train_sample)
        
        # 计算每个特征的平均SHAP值，用于判断影响方向
        mean_shap_values = np.mean(shap_values, axis=0)
        
        # 将平均SHAP值添加到特征重要性DataFrame中
        feat_imp['Mean_SHAP_Value'] = mean_shap_values
        # 根据平均SHAP值判断影响方向：正数为正向影响，负数为负向影响
        feat_imp['Impact_Direction'] = feat_imp['Mean_SHAP_Value'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
        
        # 重新保存包含SHAP信息的特征重要性文件
        feat_imp.to_csv('output/gbdt_feature_importance.csv', index=False)
        print("✅ 已更新特征重要性文件，包含SHAP影响方向")
        
        # 显示前20个重要特征的SHAP信息（仅显示Feature、Gain_Importance和Impact_Direction）
        print("\n📊 GBDT Top 20 重要特征 (含SHAP影响方向):")
        print("="*60)
        print(feat_imp[['Feature', 'Gain_Importance', 'Impact_Direction']].head(20))
        
    except Exception as e:
        print(f"⚠️ SHAP分析失败: {e}")

    # ========== Step 3: 获取叶子节点索引 ==========
    gbdt_feats_train = model.booster_.predict(x_train.values, pred_leaf=True)
    gbdt_feats_val = model.booster_.predict(x_val.values, pred_leaf=True)

    print("\n✅ GBDT 叶子节点索引 shape:", gbdt_feats_train.shape)
    print("✅ 前5个样本叶子索引:\n", gbdt_feats_train[:5])

    # ========== Step 3.5: 示例样本叶子路径 + 规则解析 ==========
    print("\n" + "="*70)
    print("🔍 解析叶子节点示例：gbdt_leaf_5_22")
    print("="*70)
    
    try:
        # 生成类别前缀列表（用于识别 one-hot 列）
        category_prefixes = [col + "_" for col in category_feature]
        
        # 解析第5棵树、第22号叶子
        leaf_rule = get_leaf_path_enhanced(
            model.booster_, 
            tree_index=5, 
            leaf_index=22, 
            feature_names=x_train.columns.tolist(),
            category_prefixes=category_prefixes
        )
        
        if leaf_rule:
            print(f"✅ 叶子节点 gbdt_leaf_5_22 的决策路径：")
            for i, rule in enumerate(leaf_rule, 1):
                print(f"   {i}. {rule}")
        else:
            print("⚠️ 未找到该叶子节点路径（可能索引越界或树结构变化）")
            
    except Exception as e:
        print("⚠️ 解析失败:", e)

    # ========== Step 4: 对叶子节点做 One-Hot 编码 ==========
    # 🆕 使用 actual_n_estimators 替代硬编码 n_estimators
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(actual_n_estimators)]

    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
    df_val_gbdt_feats = pd.DataFrame(gbdt_feats_val, columns=gbdt_feats_name)

    data_gbdt = pd.concat([df_train_gbdt_feats, df_val_gbdt_feats], ignore_index=True)

    for col in gbdt_feats_name:
        onehot_feats = pd.get_dummies(data_gbdt[col], prefix=col)
        data_gbdt.drop([col], axis=1, inplace=True)
        data_gbdt = pd.concat([data_gbdt, onehot_feats], axis=1)

    train_len = df_train_gbdt_feats.shape[0]
    train_lr = data_gbdt.iloc[:train_len, :].reset_index(drop=True)
    val_lr = data_gbdt.iloc[train_len:, :].reset_index(drop=True)

    # ========== Step 5: 训练 LR 模型 ==========
    x_train_lr, x_val_lr, y_train_lr, y_val_lr = train_test_split(
        train_lr, y_train, test_size=0.3, random_state=2018, stratify=y_train
    )

    lr = LogisticRegression(
        penalty='l2',
        C=0.1,
        solver='liblinear',
        random_state=2018,
        max_iter=1000
    )
    lr.fit(x_train_lr, y_train_lr)

    tr_logloss = log_loss(y_train_lr, lr.predict_proba(x_train_lr)[:, 1])
    val_logloss = log_loss(y_val_lr, lr.predict_proba(x_val_lr)[:, 1])
    tr_auc = roc_auc_score(y_train_lr, lr.predict_proba(x_train_lr)[:, 1])
    val_auc = roc_auc_score(y_val_lr, lr.predict_proba(x_val_lr)[:, 1])
    print('\n✅ Train LogLoss:', tr_logloss)
    print('✅ Val LogLoss:', val_logloss)
    print('✅ Train AUC:', tr_auc)
    print('✅ Val AUC:', val_auc)

    # 添加ROC曲线可视化
    fpr, tpr, _ = roc_curve(y_val_lr, lr.predict_proba(x_val_lr)[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {val_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("output/roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ ROC曲线已保存至 output/roc_curve.png")

    # ========== Step 5.5: 输出 LR 系数（哪些叶子规则最重要） ==========
    lr_coef = pd.DataFrame({
        'Leaf_Feature': x_train_lr.columns,
        'Coefficient': lr.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)

    print("\n" + "="*60)
    print("📊 LR Top 20 重要叶子特征（按系数绝对值排序）:")
    print("="*60)
    print(lr_coef.head(20))
    lr_coef.to_csv('output/lr_leaf_coefficients.csv', index=False)
    print("✅ 已保存至 output/lr_leaf_coefficients.csv")

    # ========== Step 5.6: 对高权重叶子进行规则解析 ==========
    print("\n" + "="*70)
    print("🧠 解析 LR 中高权重叶子节点对应的原始规则")
    print("="*70)
    
    top_leaves = lr_coef.head(5)  # 解析前5个最重要叶子
    category_prefixes = [col + "_" for col in category_feature]
    
    for idx, row in top_leaves.iterrows():
        leaf_feat = row['Leaf_Feature']
        coef = row['Coefficient']
        
        # 解析叶子名称，如 "gbdt_leaf_5_22"
        if leaf_feat.startswith('gbdt_leaf_'):
            parts = leaf_feat.split('_')
            if len(parts) >= 4:
                tree_idx = int(parts[2])
                leaf_idx = int(parts[3])
                
                print(f"\n🔎 解析 {leaf_feat} (LR系数: {coef:.4f})")
                try:
                    rule = get_leaf_path_enhanced(
                        model.booster_,
                        tree_index=tree_idx,
                        leaf_index=leaf_idx,
                        feature_names=x_train.columns.tolist(),
                        category_prefixes=category_prefixes
                    )
                    if rule:
                        for i, r in enumerate(rule, 1):
                            print(f"   {i}. {r}")
                    else:
                        print("   ⚠️ 路径未找到")
                except Exception as e:
                    print(f"   ⚠️ 解析失败: {e}")

    # ========== Step 6: SHAP 解释（全局 + 单样本） ==========
    print("\n" + "="*60)
    print("🎨 正在生成 SHAP 可解释性图表...")
    print("="*60)
    
    try:
        import shap
        explainer = shap.TreeExplainer(model.booster_)
        
        sample_size = min(100, len(x_val))
        x_val_sample = x_val.iloc[:sample_size]
        shap_values = explainer.shap_values(x_val_sample)

        # 1. 全局特征重要性图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, x_val_sample, feature_names=x_val_sample.columns.tolist(), show=False)
        plt.title("SHAP Feature Importance (GBDT)", fontsize=16)
        plt.tight_layout()
        plt.savefig("output/shap_summary_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ SHAP 全局特征重要性图已保存: output/shap_summary_plot.png")

        # 2. 单样本瀑布图（第0个样本）
        if len(shap_values) > 0:
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=x_val_sample.iloc[0],
                    feature_names=x_val_sample.columns.tolist()
                ),
                show=False
            )
            plt.title("SHAP Waterfall Plot - Sample 0", fontsize=16)
            plt.tight_layout()
            plt.savefig("output/shap_waterfall_sample_0.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ SHAP 单样本瀑布图已保存: output/shap_waterfall_sample_0.png")

    except Exception as e:
        print("⚠️ SHAP 解释失败（请确保已安装 shap）:", e)

    # ========== Step 7: 保存模型和必要信息用于 API ==========
    joblib.dump(model, 'output/gbdt_model.pkl')
    joblib.dump(lr, 'output/lr_model.pkl')
    
    # 🆕 保存实际树数量，供 API 使用
    pd.Series([actual_n_estimators]).to_csv('output/actual_n_estimators.csv', index=False, header=['n_estimators'])
    
    pd.Series(x_train.columns).to_csv('output/train_feature_names.csv', index=False, header=['feature'])
    pd.Series(category_feature).to_csv('output/category_features.csv', index=False, header=['feature'])
    pd.Series(continuous_feature).to_csv('output/continuous_features.csv', index=False, header=['feature'])
    
    print("✅ 模型和元数据已保存，可用于 API 服务")

    return model, lr


# ========== 主程序入口 ==========
if __name__ == '__main__':
    print("🚀 开始数据预处理...")
    data = preProcess()

    # ========== 从配置文件读取特征定义 ==========
    print("📂 正在加载特征配置...")
    feature_config = pd.read_csv('config/features.csv')
    continuous_feature = feature_config[feature_config['feature_type'] == 'continuous']['feature_name'].tolist()
    category_feature = feature_config[feature_config['feature_type'] == 'category']['feature_name'].tolist()

    print("✅ 连续特征:", continuous_feature)
    print("✅ 类别特征:", category_feature)

    print("\n✅ ======================================")
    print("✅ 将下面的内容复制到大模型内进行解读（不包括此三行）")
    print("✅ ======================================\n")

    print("针对以下(__)模型训练日志进行业务解读，输出业务规则报告，最大化创造业务价值。")

    print("🧠 开始训练 GBDT + LR 模型...")
    model, lr = gbdt_lr_train(data, category_feature, continuous_feature)

    print("\n✅ 模型训练完成！")
    print("📊 所有可解释性报告已生成在 output/ 目录下：")
    print("   - gbdt_feature_importance.csv")
    print("   - lr_leaf_coefficients.csv")
    print("   - shap_summary_plot.png")
    print("   - shap_waterfall_sample_0.png")
    print("   - actual_n_estimators.csv") 
