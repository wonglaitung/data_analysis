import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 深度学习相关导入
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: 未安装PyTorch，将跳过深度学习模型相关功能")

# 尝试导入BaseModelProcessor，可能依赖PyTorch
try:
    from base.base_model_processor import BaseModelProcessor
except ImportError as e:
    if not HAS_TORCH:
        print(f"警告: 未安装PyTorch，BaseModelProcessor导入失败: {e}")
        BaseModelProcessor = None
    else:
        raise

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
    # 检查BaseModelProcessor是否可用
    if BaseModelProcessor is None:
        print("❌ BaseModelProcessor不可用，无法进行深度学习训练")
        return
    
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
    if not HAS_TORCH:
        print("❌ 未安装PyTorch，无法训练深度学习模型")
        return None
        
    x_train_tensor = torch.FloatTensor(x_train.values)
    x_val_tensor = torch.FloatTensor(x_val.values)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
    y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1)

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # ========== Step 2: 初始化深度学习模型 ==========
    input_dim = x_train.shape[1]
    model = processor.DeepLearningModel(input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3)
    
    # 定义损失函数和优化器
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

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
    tr_ks = processor.calculate_ks_statistic(y_train, tr_pred_prob)
    val_ks = processor.calculate_ks_statistic(y_val, val_pred_prob)
    
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
    processor.plot_training_curves(train_losses, val_losses, train_aucs, val_aucs, "output/dl_training_curves.png")
    
    # 添加ROC曲线可视化
    processor.plot_roc_curve(y_val, val_pred_prob, "output/dl_roc_curve.png")

    # ========== Step 6: 特征重要性分析 ==========
    feat_imp = processor.analyze_dl_feature_importance(model, x_val_tensor, x_train.columns, "output/dl_feature_importance.csv")

    # ========== Step 7: 保存模型信息 ==========
    # 保存模型架构信息
    model_info = {
        'input_dim': input_dim,
        'hidden_dims': [256, 128, 64],
        'dropout_rate': 0.3,
        'best_val_loss': best_val_loss,
        'final_epoch': len(train_losses)
    }
    
    # 保存模型和相关信息
    processor.save_models(dl_model=model, dl_model_info=model_info)
    
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
    # 检查BaseModelProcessor是否可用
    if BaseModelProcessor is None:
        print("❌ BaseModelProcessor不可用，无法加载特征配置")
        exit(1)
        
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

    print("🧠 开始训练深度学习模型...")
    model = deep_learning_train(data, category_feature, continuous_feature)

    if model is not None:
        print("\n✅ 模型训练完成！")
        print("📊 所有可解释性报告已生成在 output/ 目录下：")
        print("   - dl_model_best.pth")
        print("   - dl_feature_importance.csv")
        print("   - dl_training_curves.png")
        print("   - dl_roc_curve.png")
        print("   - dl_model_info.csv")
    else:
        print("\n❌ 模型训练失败！")
