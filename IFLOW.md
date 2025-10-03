# 金融数据处理与机器学习项目

## 项目概述

这是一个用于处理金融数据的Python项目，主要功能包括：

1. 将多个Excel格式的数据集转换为用于机器学习的宽表格式
2. 使用GBDT+LR模型进行二分类预测
3. 生成模型可解释性报告（特征重要性、SHAP分析等）
4. 支持本地批量预测和API服务

## 目录结构

```
/data/data_analysis/
├── add_label.py            # 从Excel标签文件中提取标签并添加到宽表
├── convert.py              # 主要的数据转换脚本，将Excel文件转换为宽表
├── fake.py                 # 生成假的训练和测试数据的脚本
├── local_batch_predict.py  # 本地批量预测脚本
├── predictor.py            # 预测核心逻辑，供API和本地预测使用
├── train.py                # GBDT+LR模型训练和预测脚本
├── trim_excel.py           # 用于裁剪Excel文件的脚本
├── README.md               # 项目说明文档
├── IFLOW.md                # 项目上下文文档（当前文件）
├── config/                 # 配置文件目录
│   ├── features.csv        # 特征配置文件，定义特征类型
│   ├── primary_key.csv     # 主键配置文件，定义各Excel文件的主键字段
│   └── category_type.csv   # 类别特征类型配置文件
├── data/                   # 存放原始数据文件
│   ├── train.csv           # 训练数据集
│   ├── test.csv            # 测试数据集
│   ├── predicted_test.csv  # 本地批量预测结果
│   ├── data.csv            # 预处理后的训练和测试数据合并文件
│   └── *.xlsx              # 原始Excel数据文件
├── label/                  # 存放标签文件
│   └── *.xlsx              # 包含真实标签的Excel文件
└── output/                 # 存放处理后的输出文件和模型
    ├── feature_dict_*.csv              # 各Excel文件的字段描述字典
    ├── wide_table_*.csv                # 各Excel文件生成的宽表
    ├── feature_dictionary_global.csv   # 全局字段描述字典
    ├── ml_wide_table_global.csv        # 用于机器学习的全局宽表
    ├── ml_wide_table_with_label.csv    # 带标签的全局宽表
    ├── gbdt_feature_importance.csv     # GBDT模型特征重要性
    ├── lr_leaf_coefficients.csv        # LR模型叶子节点系数
    ├── shap_summary_plot.png           # SHAP特征重要性图
    ├── shap_waterfall_sample_0.png     # SHAP单样本瀑布图
    ├── roc_curve.png                   # ROC曲线图
    ├── submission_gbdt_lr.csv          # 模型预测结果
    ├── gbdt_model.pkl                  # GBDT模型文件
    ├── lr_model.pkl                    # LR模型文件
    ├── actual_n_estimators.csv         # 实际训练的树数量
    ├── train_feature_names.csv         # 训练特征名称
    ├── category_features.csv           # 类别特征名称
    └── continuous_features.csv         # 连续特征名称
```

## 核心功能

### 1. 数据处理与转换 (convert.py)

- 读取`data/`目录下的所有Excel文件
- 自动学习各文件的字段，分析各种维度
- 根据各种维度对数据进行透视，生成用于机器学习的宽表
- 自动计算一些衍生特征（如总和、均值、标准差等）
- 将每个文件的宽表和字段描述分别保存为CSV文件到`output/`目录中
- 合并所有文件宽表生成全局宽表`ml_wide_table_global.csv`
- 将过滤后的特征字典保存到`config/features.csv`供模型训练使用

### 2. 标签添加 (add_label.py)

- 从`label/`目录下的Excel文件中读取真实标签
- 将标签添加到全局宽表`ml_wide_table_global.csv`中
- 生成带标签的宽表`ml_wide_table_with_label.csv`
- 同时更新`data/train.csv`和`data/test.csv`文件用于模型训练

### 3. 数据裁剪 (trim_excel.py)

- 读取`data/`目录下的Excel文件
- 保留每个文件的前100条记录
- 将处理后的数据保存回原文件

### 4. 假数据生成 (fake.py)

- 读取`output/ml_wide_table_global.csv`文件
- 在Id后面增加Label字段，前1000个样本标记为1，其余为0
- 保存到`data/train.csv`
- 复制前100条数据到`data/test.csv`，并删除Label字段

### 5. 模型训练与预测 (train.py)

- 从`config/features.csv`读取特征定义
- 对类别特征进行One-Hot编码
- 训练GBDT模型并获取叶子节点索引
- 对叶子节点进行One-Hot编码
- 训练LR模型进行最终预测
- 生成模型可解释性报告（特征重要性、SHAP分析、叶子节点规则解析等）
- 保存模型和相关元数据用于API服务
- 生成ROC曲线图

### 6. 本地批量预测 (local_batch_predict.py)

- 使用训练好的模型对本地CSV文件进行批量预测
- 生成包含预测概率和解释信息的结果文件
- 结果保存在`data/predicted_*.csv`文件中

### 7. 预测核心逻辑 (predictor.py)

- 提供模型加载、数据预处理和预测的核心功能
- 支持单样本预测和批处理预测
- 生成SHAP解释和决策路径规则

## 数据说明

- `train.csv`: 训练数据集，包含标签字段Label
- `test.csv`: 测试数据集，不包含标签字段
- `*.xlsx`: 原始的Excel数据集，包含各种金融业务相关的数据
- `ml_wide_table_global.csv`: 经过处理后生成的全局宽表，用于机器学习
- `ml_wide_table_with_label.csv`: 带有真实标签的全局宽表
- `feature_dictionary_global.csv`: 全局宽表中各字段的描述信息
- `features.csv`: 特征配置文件，定义特征类型（连续/类别）
- `gbdt_model.pkl`和`lr_model.pkl`: 训练好的GBDT和LR模型
- `submission_gbdt_lr.csv`: 模型预测结果文件

## 使用说明

### 运行数据转换

```bash
python convert.py
```

该脚本会处理`data/`目录下的所有Excel文件，并在`output/`目录生成各文件的宽表和特征字典，以及全局宽表`ml_wide_table_global.csv`和`feature_dictionary_global.csv`，同时将过滤后的特征字典保存到`config/features.csv`。

### 添加真实标签

```bash
python add_label.py
```

该脚本会从`label/`目录下的Excel文件中读取真实标签，并添加到全局宽表中，生成带标签的宽表，同时更新`data/train.csv`和`data/test.csv`文件。

### 裁剪Excel文件

```bash
python trim_excel.py
```

该脚本会将`data/`目录下的Excel文件裁剪为仅包含前100条记录。

### 生成假的训练和测试数据

```bash
python fake.py
```

该脚本会读取`output/ml_wide_table_global.csv`文件，生成`data/train.csv`和`data/test.csv`。

### 训练模型

```bash
python train.py
```

该脚本会从`config/features.csv`读取特征定义，训练GBDT+LR模型，并生成预测结果和可解释性报告。

### 本地批量预测

```bash
python local_batch_predict.py <input_csv>
```

该脚本会使用训练好的模型对指定的CSV文件进行批量预测，结果保存在`data/predicted_*.csv`文件中。

## 开发约定

- 使用Python 3和pandas进行数据处理
- 使用UTF-8编码保存CSV文件
- 代码中包含详细的中文注释
- 使用`psutil`和`gc`库进行内存管理
- 使用LightGBM进行GBDT模型训练
- 使用scikit-learn进行LR模型训练
- 使用SHAP进行模型可解释性分析