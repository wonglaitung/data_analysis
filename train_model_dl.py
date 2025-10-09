import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import platform
from base.base_model_processor import BaseModelProcessor

# 仅在Windows系统上设置中文字体
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows 微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ========== 深度学习模型定义 ==========
class DeepLearningModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.3):
        super(DeepLearningModel, self).__init__()
        
        # 输入层
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        self.hidden_drops = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.hidden_drops.append(nn.Dropout(dropout_rate))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 输入层
        x = F.relu(self.input_bn(self.input_layer(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        
        # 隐藏层
        for layer, bn, drop in zip(self.hidden_layers, self.hidden_bns, self.hidden_drops):
            x = F.relu(bn(layer(x)))
            x = drop(x)
        
        # 输出层
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        return x

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
    
    # 确保所有列都是数值类型，将非数值类型转换为数值类型
    for col in data.columns:
        if data[col].dtype == 'object':
            # 尝试转换为数值类型，无法转换的设置为-1
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(-1)
    
    data.to_csv('data_train/data.csv', index=False, encoding='utf-8')
    return data

# ========== 深度学习训练函数 ==========
def deep_learning_train(data, category_feature, continuous_feature):
    """
    使用深度学习训练模型，增强可解释性输出
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

    # 确保所有特征都是数值类型
    for col in train.columns:
        if train[col].dtype == 'object':
            train[col] = pd.to_numeric(train[col], errors='coerce').fillna(-1)
        # 确保数据类型为float32，避免类型转换错误
        train[col] = pd.to_numeric(train[col], errors='coerce').fillna(-1).astype('float32')
    
    # 再次检查并确保没有object类型的数据
    object_cols = train.dtypes[train.dtypes == 'object'].index.tolist()
    if object_cols:
        for col in object_cols:
            train[col] = pd.to_numeric(train[col], errors='coerce').fillna(-1).astype('float32')

    # 划分训练/验证集
    # 确保标签也是数值类型
    target = pd.to_numeric(target, errors='coerce').fillna(0).astype('float32')
    
    x_train, x_val, y_train, y_val = train_test_split(
        train, target, test_size=0.2, random_state=2020, stratify=target
    )

    # 转换为PyTorch张量
    x_train_tensor = torch.FloatTensor(x_train.values)
    x_val_tensor = torch.FloatTensor(x_val.values)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
    y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1)

    # 创建数据加载器
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # ========== Step 2: 初始化深度学习模型 ==========
    input_dim = x_train.shape[1]
    model = DeepLearningModel(input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # ========== Step 3: 训练模型 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    num_epochs = 100
    best_val_loss = float('inf')
    early_stopping_patience = 15
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []
    
    print("🚀 开始深度学习模型训练...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(batch_y.cpu().numpy())
        
        train_loss /= len(train_loader.dataset)
        train_auc = roc_auc_score(np.array(train_targets), np.array(train_preds))
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_auc = roc_auc_score(np.array(val_targets), np.array(val_preds))
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录损失和AUC
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        
        # 打印进度
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'output/dl_model_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ========== Step 4: 加载最佳模型并评估 ==========
    model.load_state_dict(torch.load('output/dl_model_best.pth'))
    model.eval()
    
    # 计算最终预测结果
    with torch.no_grad():
        tr_pred_prob = model(x_train_tensor.to(device)).cpu().numpy().flatten()
        val_pred_prob = model(x_val_tensor.to(device)).cpu().numpy().flatten()
    
    tr_logloss = log_loss(y_train, tr_pred_prob)
    val_logloss = log_loss(y_val, val_pred_prob)
    
    # 计算 KS 统计量
    def calculate_ks_statistic(y_true, y_pred_prob):
        from scipy.stats import ks_2samp
        # 将样本按预测概率排序
        data = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob})
        data_sorted = data.sort_values('y_pred_prob', ascending=False)
        
        # 计算累积分布
        cum_positive = (data_sorted['y_true'] == 1).cumsum() / (y_true == 1).sum()
        cum_negative = (data_sorted['y_true'] == 0).cumsum() / (y_true == 0).sum()
        
        # KS统计量是两个累积分布之间的最大差异
        ks_stat = np.max(np.abs(cum_positive - cum_negative))
        return ks_stat
    
    tr_ks = calculate_ks_statistic(y_train, tr_pred_prob)
    val_ks = calculate_ks_statistic(y_val, val_pred_prob)
    
    tr_auc = roc_auc_score(y_train, tr_pred_prob)
    val_auc = roc_auc_score(y_val, val_pred_prob)
    
    print('\n✅ Train LogLoss:', tr_logloss)
    print('✅ Val LogLoss:', val_logloss)
    print('✅ Train KS:', tr_ks)
    print('✅ Val KS:', val_ks)
    print('✅ Train AUC:', tr_auc)
    print('✅ Val AUC:', val_auc)

    # ========== Step 5: 可视化结果 ==========
    # 绘制损失曲线
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
    plt.savefig("output/dl_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 训练曲线已保存至 output/dl_training_curves.png")
    
    # 添加ROC曲线可视化
    fpr, tpr, _ = roc_curve(y_val, val_pred_prob)
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
    plt.savefig("output/dl_roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ ROC曲线已保存至 output/dl_roc_curve.png")

    # ========== Step 6: 特征重要性分析 ==========
    print("\n" + "="*60)
    print("🧠 正在分析特征重要性...")
    print("="*60)
    
    # 使用梯度方法计算特征重要性（仅在CPU上计算以避免GPU内存问题）
    model.eval()
    x_sample = x_val_tensor[:500].to("cpu")  # 使用更少的样本计算重要性
    x_sample.requires_grad = True
    
    output = model(x_sample)
    output.sum().backward()
    
    # 计算特征重要性（梯度的绝对值）
    feature_importance = torch.abs(x_sample.grad).mean(dim=0).cpu().numpy()
    
    # 创建特征重要性DataFrame
    feat_imp = pd.DataFrame({
        'Feature': x_train.columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # ========== 增加：通过梯度符号分析特征影响方向 ==========
    try:
        print("\n" + "="*60)
        print("🧠 正在通过梯度符号分析特征影响方向...")
        print("="*60)
        
        # 计算每个特征的平均梯度值，用于判断影响方向
        mean_grad_values = x_sample.grad.mean(dim=0).cpu().numpy()
        
        # 将平均梯度值添加到特征重要性DataFrame中
        feat_imp['Mean_Grad_Value'] = mean_grad_values
        # 根据平均梯度值判断影响方向：正数为正向影响，负数为负向影响
        feat_imp['Impact_Direction'] = feat_imp['Mean_Grad_Value'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
        
        # 保存包含所有信息的特征重要性文件
        feat_imp.to_csv('output/dl_feature_importance.csv', index=False)
        print("✅ 特征重要性已保存至 output/dl_feature_importance.csv")
        
        # 显示前20个重要特征的完整信息
        print("\n📊 深度学习模型 Top 20 重要特征 (含影响方向):")
        print("="*60)
        print(feat_imp[['Feature', 'Importance', 'Impact_Direction']].head(20))
        
    except Exception as e:
        print(f"⚠️ 特征影响方向分析失败: {e}")
        # 如果分析失败，仍保留基本的特征重要性信息
        feat_imp['Impact_Direction'] = 'Unknown'
        # 保存包含所有信息的特征重要性文件
        feat_imp.to_csv('output/dl_feature_importance.csv', index=False)
        print("✅ 特征重要性已保存至 output/dl_feature_importance.csv")
        
        # 显示前20个重要特征
        print("\n📊 深度学习模型 Top 20 重要特征:")
        print("="*60)
        print(feat_imp.head(20))

    # ========== Step 7: 保存模型信息 ==========
    # 保存模型架构信息
    model_info = {
        'input_dim': input_dim,
        'hidden_dims': [256, 128, 64],
        'dropout_rate': 0.3,
        'best_val_loss': best_val_loss,
        'final_epoch': len(train_losses)
    }
    
    pd.DataFrame([model_info]).to_csv('output/dl_model_info.csv', index=False)
    print("✅ 模型信息已保存至 output/dl_model_info.csv")

    print("✅ 深度学习模型训练完成！")
    print("📊 所有可解释性报告已生成在 output/ 目录下：")
    print("   - dl_model_best.pth")
    print("   - dl_feature_importance.csv")
    print("   - dl_training_curves.png")
    print("   - dl_roc_curve.png")
    print("   - dl_model_info.csv")

    return model

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

    print("对以下(推荐/授信/预警)模型训练日志进行分析，输出银行业务人员可以理解的解读报告，目地是进行(推荐/授信/预警)，通过模型分析赋能业务决策。\n")

    print("🧠 开始训练深度学习模型...")
    model = deep_learning_train(data, category_feature, continuous_feature)

    print("\n✅ 模型训练完成！")
    print("📊 所有可解释性报告已生成在 output/ 目录下：")
    print("   - dl_model_best.pth")
    print("   - dl_feature_importance.csv")
    print("   - dl_training_curves.png")
    print("   - dl_roc_curve.png")
    print("   - dl_model_info.csv")
