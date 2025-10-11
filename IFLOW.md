# 金融数据处理与机器学习项目

## 项目概述

这是一个用于处理金融数据的Python项目，主要功能包括：

1. 将多个Excel格式的数据集转换为用于机器学习的宽表格式
2. 使用GBDT+LR模型进行二分类训练
3. 生成模型可解释性报告（特征重要性、特征贡献分析等）
4. 对新的数据进行预测并提供解释性结果
5. 检测模型的公平性，确保对不同群体的客户公平对待
6. 支持深度学习模型训练（新增）
7. 在数据处理过程中增加数据分析功能，全面了解数据结构和质量

## 目录结构

```
/data/data_analysis/
├── add_train_label.py      # 从Excel标签文件中提取标签并添加到宽表
├── convert_train_data.py   # 主要的数据转换脚本，将Excel文件转换为宽表
├── convert_predict_data.py # 预测数据转换脚本，将Excel文件转换为宽表用于预测
├── fake_train_data.py      # 生成假的训练和测试数据的脚本
├── train_model.py          # GBDT+LR模型训练脚本（仅训练，不进行预测）
├── train_model_dl.py       # 深度学习模型训练脚本（新增）
├── predict.py              # 使用训练好的模型进行预测的脚本
├── check_model_fairness.py # 检测模型公平性的脚本
├── trim_excel.py           # 用于裁剪Excel文件的脚本
├── base/                   # 基础类目录
│   ├── base_data_processor.py  # 基础数据处理器类
│   ├── base_model_processor.py # 基础模型处理器类
│   └── data_analyzer.py    # 数据分析器类
├── README.md               # 项目说明文档
├── Business_User_Guide.md  # 业务用户使用手册
├── IFLOW.md                # 项目上下文文档（当前文件）
├── config/                 # 配置文件目录
│   ├── features.csv        # 特征配置文件，定义特征类型
│   ├── primary_key.csv     # 主键配置文件，定义各Excel文件的主键字段
│   ├── category_type.csv   # 类别特征类型配置文件
│   ├── label_key.csv       # 标签文件配置文件
│   └── sensitive_attr.csv  # 敏感属性配置文件，定义用于公平性检测的敏感属性字段
├── data_train/             # 存放训练用的原始数据文件
│   ├── train.csv           # 训练数据集
│   ├── test.csv            # 测试数据集
│   ├── data.csv            # 预处理后的训练和测试数据合并文件
│   └── *.xlsx              # 原始Excel数据文件
├── data_predict/           # 存放待预测的原始数据文件
│   └── *.xlsx              # 原始Excel数据文件
├── label_train/            # 存放标签文件
│   └── *.xlsx              # 原始标签Excel文件
└── output/                 # 存放处理后的输出文件、模型和预测结果
    ├── feature_dictionary_global.csv   # 全局字段描述字典
    ├── feature_dictionary_predict_global.csv # 预测数据全局字段描述字典
    ├── ml_wide_table_global.csv        # 用于机器学习的全局宽表
    ├── ml_wide_table_predict_global.csv # 用于预测的全局宽表
    ├── ml_wide_table_with_label.csv    # 带有真实标签的全局宽表
    ├── gbdt_feature_importance.csv     # GBDT模型特征重要性
    ├── fairness_metrics.csv            # 模型公平性指标
    ├── roc_curve.png                   # ROC曲线图
    ├── gbdt_model.pkl                  # GBDT模型文件
    ├── lr_model.pkl                    # LR模型文件
    ├── actual_n_estimators.csv         # 实际训练的树数量
    ├── train_feature_names.csv         # 训练特征名称
    ├── category_features.csv           # 类别特征名称
    ├── continuous_features.csv         # 连续特征名称
    ├── lr_leaf_coefficients.csv        # LR模型叶子节点系数
    └── prediction_results.csv          # 预测结果文件
```

## 特征类型识别机制

在数据处理过程中，程序会自动识别每个字段的特征类型（连续型、类别型或文本型）。识别机制如下：

### 1. 强制类别特征识别
程序会读取`config/category_type.csv`配置文件，该文件中指定的字段会被强制识别为类别特征，即使它们在数据上表现为数值型。这解决了业务上某些数值型字段实际上应作为类别特征处理的问题。

配置文件格式：
```
file_name,column_name,feature_type
数据集-环比-带金额_20241231.xlsx,网点号,category
数据集-同比-带金额_20241231(1).xlsx,网点号,category
数据集-同比-带金额_20241231(1).xlsx,日期,category
分渠道汇款业务分析_20241231.xlsx,BELONG_BRNO,category
```

如果`category_type.csv`文件不存在，程序会完全依赖自动特征类型识别机制。

### 2. 自动特征类型识别
对于未在`category_type.csv`中配置的字段，程序会根据以下规则自动识别特征类型：

1. **连续型特征（continuous）**：
   - 数据类型为数值型的字段（如int、float等）

2. **类别型特征（category）**：
   - 数据类型为字符串或对象类型，且唯一值数量≤10个的字段

3. **文本型特征（text）**：
   - 数据类型为字符串或对象类型，且唯一值数量>10个的字段

4. **其他类型（other）**：
   - 不符合上述任何条件的字段

这种识别机制确保了既能满足业务需求（通过强制指定），又能自动处理大部分常规情况。

## 数据安全机制（新增）

为防止敏感数据泄露，项目增加了数据安全检查机制：

1. **敏感字段检测**：在数据处理过程中自动检测敏感字段（如客户名称、身份证号、手机号、地址等）
2. **繁简体转换处理**：支持繁简体转换，确保能检测到不同语言形式的敏感字段
3. **程序自动退出**：一旦检测到敏感字段，程序会立即退出并提示用户

## 核心功能

### 1. 数据处理与转换 (convert_train_data.py)

- 读取`data_train/`目录下的所有Excel文件
- 自动学习各文件的字段，分析各种维度
- 根据各种维度对数据进行透视，生成用于机器学习的宽表
- 自动计算一些衍生特征（如总和、均值、标准差等）
- 合并所有文件宽表生成全局宽表`ml_wide_table_global.csv`
- 将过滤后的特征字典保存到`config/features.csv`供模型训练使用
- 从`config/primary_key.csv`读取各文件的主键配置
- 从`config/category_type.csv`读取需要强制作为类别特征的字段配置
- 对类别特征进行One-Hot编码并合并到最终宽表中
- 检测并防止敏感数据泄露
- 在数据处理过程中增加数据分析功能，全面了解数据结构和质量
- 不再保存每个文件的独立宽表，只保留最终大宽表

### 2. 标签添加 (add_train_label.py)

- 从`label_train/`目录下的指定Excel文件中读取标签数据（"本地支薪"字段）
- 处理ID字段的前导零问题，确保数据正确匹配
- 将标签数据合并到全局宽表中
- 生成带标签的训练数据集`data_train/train.csv`和测试数据集`data_train/test.csv`
- 处理缺失标签值，将其默认设置为0
- 从`config/label_key.csv`读取标签文件配置信息

### 3. 数据裁剪 (trim_excel.py)

- 读取`data_train/`目录下的Excel文件
- 保留每个文件的前100条记录
- 将处理后的数据保存回原文件

### 4. 假数据生成 (fake_train_data.py)

- 读取`output/ml_wide_table_global.csv`文件
- 在Id后面增加Label字段，前1000个样本标记为1，其余为0
- 保存到`data_train/train.csv`
- 复制前100条数据到`data_train/test.csv`，并删除Label字段

### 5. 模型训练 (train_model.py)

- 从`config/features.csv`读取特征定义
- 对类别特征进行One-Hot编码
- 训练GBDT模型并获取叶子节点索引
- 对叶子节点进行One-Hot编码
- 训练LR模型
- 生成模型可解释性报告（特征重要性、特征贡献分析、叶子节点规则解析等）
- 保存模型和相关元数据用于API服务
- 生成ROC曲线图
- 支持早停机制，自动确定最佳迭代次数
- 不再生成预测结果文件（submission_gbdt_lr.csv）
- 可将(推荐/授信/预警)模型训练日志放入大模型进行分析，输出银行业务人员可以理解的解读报告，通过模型分析赋能业务决策
- 计算并输出KS统计量，用于评估模型性能
- 使用LightGBM内置功能分析特征影响方向
- 保存实际训练的树数量用于预测时的一致性检查

### 6. 深度学习模型训练 (train_model_dl.py)（新增）

- 支持使用深度学习模型进行训练
- 提供与传统GBDT+LR模型的对比分析

### 7. 预测数据处理与转换 (convert_predict_data.py)

- 读取`data_predict/`目录下的所有Excel文件
- 使用与训练数据相同的处理逻辑生成宽表
- 确保特征与训练数据保持一致
- 生成用于预测的全局宽表`ml_wide_table_predict_global.csv`
- 检查预测数据特征字段与训练时使用的特征字段是否匹配
- 在数据处理过程中增加数据分析功能，全面了解数据结构和质量
- 不再保存每个文件的独立宽表，只保留最终大宽表

### 8. 使用模型进行预测 (predict.py)

- 加载训练好的GBDT和LR模型
- 加载特征配置文件
- 准备预测数据，确保特征与训练时一致
- 使用GBDT模型获取叶子节点索引
- 对叶子节点进行One-Hot编码
- 使用LR模型进行预测
- 生成预测解释性信息（重要特征、特征贡献值、决策规则等）（仅在使用--shap参数时计算特征贡献值）
- 保存预测结果到`output/prediction_results.csv`
- 支持特征贡献可视化（生成特征贡献图）

### 9. 模型公平性检测 (check_model_fairness.py)

- 加载训练好的GBDT和LR模型
- 从`config/sensitive_attr.csv`读取敏感属性配置
- 计算多种公平性指标，包括：
  - 人口统计学公平性(Demographic Parity)
  - 机会均等(Equal Opportunity)
  - 均衡几率(Equalized Odds)
  - 预测公平性(Predictive Parity)
- 生成公平性检测报告并保存到`output/fairness_metrics.csv`

## 配置文件说明

### 1. 主键配置 (config/primary_key.csv)

定义各Excel文件的主键字段，包含以下列：
- `file_name`: Excel文件名
- `sheet_name`: 工作表名（可为空）
- `primary_key`: 主键字段名

### 2. 类别特征配置 (config/category_type.csv)

定义需要强制作为类别特征的字段，包含以下列：
- `file_name`: Excel文件名
- `column_name`: 列名
- `feature_type`: 特征类型（固定为category）

### 3. 标签配置 (config/label_key.csv)

定义标签文件的信息，包含以下列：
- `file_name`: 标签Excel文件名
- `sheet_name`: 工作表名
- `label_key`: 标签列名

### 4. 敏感属性配置 (config/sensitive_attr.csv)

定义用于公平性检测的敏感属性字段，包含以下列：
- `file_name`: 包含敏感属性的Excel文件名
- `sheet_name`: 工作表名（可为空）
- `column_name`: 敏感属性列名

配置示例：
```
file_name,sheet_name,column_name
分渠道汇款业务分析_20241231.xlsx,,BELONG_BRNO
```

### 5. 特征配置 (config/features.csv)

定义所有特征的类型，包含以下列：
- `feature_name`: 特征名称
- `feature_type`: 特征类型（continuous/category）

## 数据说明

- `data_train/train.csv`: 训练数据集，包含标签字段Label
- `data_train/test.csv`: 测试数据集，不包含标签字段
- `data_train/*.xlsx`: 原始的Excel数据集，包含各种金融业务相关的数据
- `data_predict/*.xlsx`: 待预测的原始Excel数据集
- `label_train/*.xlsx`: 原始标签Excel文件
- `ml_wide_table_global.csv`: 经过处理后生成的全局宽表，用于机器学习建模
- `ml_wide_table_predict_global.csv`: 经过处理后生成的全局宽表，用于预测
- `ml_wide_table_with_label.csv`: 带有真实标签的全局宽表
- `feature_dictionary_global.csv`: 全局宽表中各字段的描述信息
- `features.csv`: 特征配置文件，定义特征类型（连续/类别）
- `gbdt_model.pkl`和`lr_model.pkl`: 训练好的GBDT和LR模型
- `prediction_results.csv`: 预测结果文件，包含样本ID、预测概率和解释性信息
- `fairness_metrics.csv`: 模型公平性指标文件，包含各种公平性指标的得分

## 模型评估结果

### 模型性能指标
- **训练集 LogLoss**: 0.0215
- **验证集 LogLoss**: 0.0246
- **训练集 AUC**: 0.9779
- **验证集 AUC**: 0.9819
- **训练集 KS**: 未提供
- **验证集 KS**: 未提供

### 结果解释
- LogLoss 非常小，说明模型预测概率非常接近真实标签
- AUC 超过 0.97，说明模型具有很好的区分能力
- 模型没有明显过拟合，具有良好的泛化能力

## 公平性检测结果

### 公平性指标说明
- **Demographic Parity (人口统计学公平性)**: 衡量模型对不同群体的预测正类概率是否相近
- **Equal Opportunity (机会均等)**: 衡量模型对不同群体中真正例的识别能力是否相近
- **Equalized Odds (均衡几率)**: 综合衡量模型对不同群体的真阳性率和假阳性率是否相近
- **Predictive Parity (预测公平性)**: 衡量模型对不同群体的预测准确性是否相近

所有指标得分范围为0-1，越接近1表示模型越公平。

### 实际检测结果
```
Metric,Score
Demographic Parity,0.9996
Equal Opportunity,0.9474
Equalized Odds,0.9737
Predictive Parity,0.0
```

## 使用说明

### 建模阶段

1. **运行数据转换**
```bash
python convert_train_data.py
```
该脚本会处理`data_train/`目录下的所有Excel文件，并在`output/`目录生成全局宽表`ml_wide_table_global.csv`和`feature_dictionary_global.csv`，同时将过滤后的特征字典保存到`config/features.csv`。数据分析结果会显示在屏幕上但不会保存到文件。

2. **添加真实标签**
```bash
python add_train_label.py
```
该脚本会从`label_train/`目录下的指定Excel文件中提取标签数据并合并到全局宽表中，生成带标签的训练数据集和测试数据集。

3. **训练模型**
```bash
python train_model.py
```
该脚本会从`config/features.csv`读取特征定义，训练GBDT+LR模型，并保存模型文件和相关元数据。

或者使用深度学习模型训练：
```bash
python train_model_dl.py
```

### 预测阶段

1. **运行预测数据转换**
```bash
python convert_predict_data.py
```
该脚本会处理`data_predict/`目录下的所有Excel文件，并生成用于预测的全局宽表`ml_wide_table_predict_global.csv`。数据分析结果会显示在屏幕上但不会保存到文件。

2. **使用模型进行预测**
```bash
# 仅进行预测（默认，速度较快）
python predict.py

# 进行预测并计算特征贡献值（较慢但提供可解释性分析）
python predict.py --shap
```
该脚本会加载训练好的模型，对预测数据进行预测，并将结果保存到`output/prediction_results.csv`。默认情况下只进行预测，不计算特征贡献值以提高速度。如果需要模型决策的可解释性分析，可以添加`--shap`参数来计算特征贡献值。

3. **检测模型公平性**
```bash
python check_model_fairness.py
```
该脚本会加载训练好的模型，从`config/sensitive_attr.csv`读取敏感属性配置，计算多种公平性指标，并将结果保存到`output/fairness_metrics.csv`。

## 预测结果说明

预测结果保存在`output/prediction_results.csv`文件中，包含以下列：
- `Id`: 样本ID
- `PredictedProb`: 预测概率（0-1之间）
- `top_features`: 前3个最重要的特征及其特征贡献值（仅在使用--shap参数时计算）
- `top_rules`: 前3个决策规则（仅在使用--shap参数时计算）

## 公平性检测结果说明

公平性检测结果保存在`output/fairness_metrics.csv`文件中，包含以下列：
- `Metric`: 公平性指标名称
- `Score`: 公平性得分（0-1之间，越接近1表示越公平）

## 开发约定

- 使用Python 3和pandas进行数据处理
- 使用UTF-8编码保存CSV文件
- 代码中包含详细的中文注释
- 使用`psutil`和`gc`库进行内存管理
- 使用LightGBM进行GBDT模型训练
- 使用scikit-learn进行LR模型训练
- 使用LightGBM内置功能进行模型可解释性分析
- 增加了数据安全检查，防止敏感数据泄露