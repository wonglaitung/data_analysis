import os
#os.environ["NUMBA_DISABLE_TBB"] = "1"
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import log_evaluation
import matplotlib.pyplot as plt
import platform
from base.base_model_processor import BaseModelProcessor

# 仅在Windows系统上设置中文字体
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows 微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


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
    processor = BaseModelProcessor()
    
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

    # ========== Step 2.5: 输出 GBDT 特征重要性（含影响方向） ==========
    # 使用基类中的方法分析特征重要性
    feat_imp = processor.analyze_feature_importance(model.booster_, x_train.columns.tolist())
    
    # ========== 增加：通过LightGBM内置功能分析特征影响方向 ==========
    try:
        # 获取训练集样本的特征贡献值
        contrib_values = model.booster_.predict(x_train.values, pred_contrib=True)
        
        # contrib_values的形状为 (n_samples, n_features + 1)
        # 最后一列是期望值（base value），前面的列是各特征的贡献值
        
        # 计算每个特征的平均贡献值，用于判断影响方向
        mean_contrib_values = np.mean(contrib_values[:, :-1], axis=0)  # 排除最后一列期望值
        
        # 将平均贡献值添加到特征重要性DataFrame中
        feat_imp['Mean_Contrib_Value'] = mean_contrib_values
        # 根据平均贡献值判断影响方向：正数为正向影响，负数为负向影响
        feat_imp['Impact_Direction'] = feat_imp['Mean_Contrib_Value'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
        
        # 保存包含所有信息的特征重要性文件
        feat_imp.to_csv('output/gbdt_feature_importance.csv', index=False)
        print("✅ 已保存特征重要性文件至 output/gbdt_feature_importance.csv")
        
        # 显示前20个重要特征的完整信息（不重复打印标题）
        print("\n" + "="*60)
        print("📊 GBDT Top 20 重要特征 (含影响方向):")
        print("="*60)
        print(feat_imp[['Feature', 'Gain_Importance', 'Split_Importance', 'Impact_Direction']].head(20))
        
    except Exception as e:
        print(f"⚠️ 特征贡献分析失败: {e}")
        # 如果分析失败，仍保留基本的特征重要性信息
        feat_imp['Impact_Direction'] = 'Unknown'

    # ========== Step 3: 获取叶子节点索引 ==========
    gbdt_feats_train = model.booster_.predict(x_train.values, pred_leaf=True)
    gbdt_feats_val = model.booster_.predict(x_val.values, pred_leaf=True)

    # 不再输出叶子节点索引的详细信息

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

    # 计算训练集和验证集的预测概率
    tr_pred_prob = lr.predict_proba(x_train_lr)[:, 1]
    val_pred_prob = lr.predict_proba(x_val_lr)[:, 1]

    tr_logloss = log_loss(y_train_lr, tr_pred_prob)
    val_logloss = log_loss(y_val_lr, val_pred_prob)
    
    # 计算 KS 统计量
    tr_ks = processor.calculate_ks_statistic(y_train_lr, tr_pred_prob)
    val_ks = processor.calculate_ks_statistic(y_val_lr, val_pred_prob)
    
    tr_auc = roc_auc_score(y_train_lr, tr_pred_prob)
    val_auc = roc_auc_score(y_val_lr, val_pred_prob)
    print('\n✅ Train LogLoss:', tr_logloss)
    print('✅ Val LogLoss:', val_logloss)
    print('✅ Train KS:', tr_ks)
    print('✅ Val KS:', val_ks)
    print('✅ Train AUC:', tr_auc)
    print('✅ Val AUC:', val_auc)

    # 添加ROC曲线可视化
    processor.plot_roc_curve(y_val_lr, lr.predict_proba(x_val_lr)[:, 1], "output/roc_curve.png")

    # ========== Step 5.5: 输出 LR 系数（哪些叶子规则最重要） ==========
    lr_coef = pd.DataFrame({
        'Leaf_Feature': x_train_lr.columns,
        'Coefficient': lr.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)

    print("\n" + "="*60)
    print("📊 LR Top 20 重要叶子特征（按系数绝对值排序）:")
    #print("="*60)
    #print(lr_coef.head(20))
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
                    rule = processor.get_leaf_path_enhanced(
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

    # ========== Step 6: 特征贡献可视化 ==========
    print("\n" + "="*60)
    print("🎨 正在生成特征贡献可视化图表...")
    print("="*60)
    
    # 使用LightGBM内置的特征贡献计算
    print("ℹ️  已使用LightGBM内置功能计算特征贡献")

    # 保存模型和相关配置
    processor.save_models(model, lr, category_feature, continuous_feature)
    
    print("✅ 模型训练完成！")
    print("📊 所有可解释性报告已生成在 output/ 目录下：")
    print("   - gbdt_feature_importance.csv")
    print("   - lr_leaf_coefficients.csv")
    print("   - actual_n_estimators.csv")
    print("   - train_feature_names.csv")
    print("   - category_features.csv")
    print("   - continuous_features.csv")

    return model, lr


# ========== 主程序入口 ==========
if __name__ == '__main__':
    print("🚀 开始数据预处理...")
    data = preProcess()

    # ========== 从配置文件读取特征定义 ==========
    print("📂 正在加载特征配置...")
    processor = BaseModelProcessor()
    if not processor.load_feature_config():
        print("❌ 加载特征配置失败")
        exit(1)
        
    continuous_feature = processor.continuous_features
    category_feature = processor.category_features

    print("✅ 连续特征:", continuous_feature)
    print("✅ 类别特征:", category_feature)

    # 显示大模型解读提示
    processor.show_model_interpretation_prompt()

    print("🧠 开始训练 GBDT + LR 模型...")
    model, lr = gbdt_lr_train(data, category_feature, continuous_feature)

    print("\n✅ 模型训练完成！")
    print("📊 所有可解释性报告已生成在 output/ 目录下：")
    print("   - gbdt_feature_importance.csv")
    print("   - lr_leaf_coefficients.csv")
    print("   - actual_n_estimators.csv") 
