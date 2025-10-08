#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型公平性检测工具
用于在模型训练后检测公平性和偏见
"""

import pandas as pd
import numpy as np
import os
import sys
from base.base_fairness_processor import FairnessChecker, identify_potential_sensitive_features

def load_training_data_with_predictions():
    """
    加载训练数据和模型预测结果
    
    返回:
    DataFrame: 包含标签和预测结果的数据框
    """
    try:
        # 加载训练数据
        train_data_path = "data_train/train.csv"
        if not os.path.exists(train_data_path):
            print(f"❌ 找不到训练数据文件: {train_data_path}")
            return None
            
        print(f"正在加载训练数据: {train_data_path}")
        train_df = pd.read_csv(train_data_path)
        print(f"✅ 训练数据加载完成，形状: {train_df.shape}")
        
        # 检查是否包含Label列
        if 'Label' not in train_df.columns:
            print("❌ 训练数据中未找到Label列")
            return None
            
        return train_df
        
    except Exception as e:
        print(f"❌ 加载训练数据时出错: {e}")
        return None

def load_prediction_results():
    """
    加载预测结果
    
    返回:
    DataFrame: 包含预测结果的数据框
    """
    try:
        # 加载预测结果
        prediction_path = "output/prediction_results.csv"
        if not os.path.exists(prediction_path):
            print(f"❌ 找不到预测结果文件: {prediction_path}")
            return None
            
        print(f"正在加载预测结果: {prediction_path}")
        pred_df = pd.read_csv(prediction_path)
        print(f"✅ 预测结果加载完成，形状: {pred_df.shape}")
        
        # 检查必需的列
        required_columns = ['Id', 'PredictedProb']
        missing_columns = [col for col in required_columns if col not in pred_df.columns]
        if missing_columns:
            print(f"❌ 预测结果中缺少必需的列: {missing_columns}")
            return None
            
        return pred_df
        
    except Exception as e:
        print(f"❌ 加载预测结果时出错: {e}")
        return None

def perform_fairness_analysis():
    """
    执行公平性分析
    """
    print("=== 模型公平性检测 ===")
    
    # 加载训练数据
    train_df = load_training_data_with_predictions()
    if train_df is None:
        return False
    
    # 加载预测结果
    pred_df = load_prediction_results()
    if pred_df is None:
        return False
    
    # 合并数据
    print("正在合并训练数据和预测结果...")
    merged_df = pd.merge(train_df[['Id', 'Label']], pred_df[['Id', 'PredictedProb']], on='Id', how='inner')
    print(f"✅ 数据合并完成，形状: {merged_df.shape}")
    
    if len(merged_df) == 0:
        print("❌ 合并后的数据为空")
        return False
    
    # 识别潜在的敏感特征
    print("\n=== 识别潜在敏感特征 ===")
    sensitive_features = identify_potential_sensitive_features(train_df)
    print(f"🔍 识别到的潜在敏感特征: {sensitive_features}")
    
    # 如果没有找到敏感特征，使用一些默认的特征进行分析
    if not sensitive_features:
        # 查找可能的分组特征
        potential_group_features = []
        for col in train_df.columns:
            # 查找类别型特征或具有有限唯一值的特征
            if col not in ['Id', 'Label'] and train_df[col].dtype == 'object':
                unique_count = train_df[col].nunique()
                if 2 <= unique_count <= 20:  # 限制在合理范围内
                    potential_group_features.append(col)
            elif col not in ['Id', 'Label'] and train_df[col].dtype in ['int64', 'float64']:
                unique_count = train_df[col].nunique()
                if 2 <= unique_count <= 10:  # 对于数值型，限制在更小范围内
                    potential_group_features.append(col)
        
        sensitive_features = potential_group_features[:5]  # 取前5个
        print(f"🔄 使用默认分组特征: {sensitive_features}")
    
    if not sensitive_features:
        print("⚠️ 未找到任何潜在的敏感特征，将使用Id的首字符作为示例分组")
        # 创建一个示例分组特征
        train_df['example_group'] = train_df['Id'].astype(str).str[0]
        sensitive_features = ['example_group']
    
    # 创建公平性检测器
    fairness_checker = FairnessChecker(merged_df, label_col='Label', prediction_col='PredictedProb')
    
    # 生成公平性报告
    print(f"\n=== 执行公平性检测 ===")
    fairness_report = fairness_checker.generate_fairness_report(
        protected_attrs=sensitive_features,
        threshold=0.5,
        fairness_threshold=0.1  # 10%的差异阈值
    )
    
    # 保存报告
    report_path = "output/fairness_report.csv"
    save_fairness_report(fairness_report, report_path)
    
    # 打印摘要
    print_fairness_summary(fairness_report)
    
    return True

def save_fairness_report(report, file_path):
    """
    保存公平性报告到CSV文件
    
    参数:
    report: 公平性报告字典
    file_path: 保存路径
    """
    try:
        # 将报告转换为DataFrame格式
        rows = []
        
        for attr, attr_report in report.items():
            if 'error' in attr_report:
                rows.append({
                    'protected_attribute': attr,
                    'error': attr_report['error']
                })
                continue
                
            # 为每个公平性指标创建一行
            metrics = ['demographic_parity', 'equal_opportunity', 'equalized_odds', 'predictive_parity']
            for metric in metrics:
                if metric in attr_report:
                    metric_data = attr_report[metric]
                    row = {
                        'protected_attribute': attr,
                        'metric': metric,
                        'max_difference': metric_data.get('max_difference', 
                                                       metric_data.get('max_tpr_difference', 0) or 
                                                       metric_data.get('max_fpr_difference', 0)),
                        'bias_detected': attr_report['bias_detected'].get(f'{metric}_bias', False),
                        'overall_bias': attr_report.get('overall_bias', False)
                    }
                    rows.append(row)
        
        if rows:
            report_df = pd.DataFrame(rows)
            report_df.to_csv(file_path, index=False, encoding='utf-8')
            print(f"✅ 公平性报告已保存到: {file_path}")
        else:
            print("⚠️ 没有数据可保存到公平性报告")
            
    except Exception as e:
        print(f"❌ 保存公平性报告时出错: {e}")

def print_fairness_summary(report):
    """
    打印公平性检测摘要
    
    参数:
    report: 公平性报告字典
    """
    print("\n=== 公平性检测摘要 ===")
    
    overall_bias_found = False
    
    for attr, attr_report in report.items():
        if 'error' in attr_report:
            print(f"❌ 属性 '{attr}': {attr_report['error']}")
            continue
            
        print(f"\n🔍 属性 '{attr}':")
        
        # 检查是否发现偏见
        if attr_report.get('overall_bias', False):
            print(f"  ⚠️  发现偏见!")
            overall_bias_found = True
        else:
            print(f"  ✅ 未发现明显偏见")
        
        # 打印各指标的最大差异
        metrics_info = {
            'demographic_parity': '人口统计学公平性',
            'equal_opportunity': '机会平等',
            'equalized_odds': '均衡机会',
            'predictive_parity': '预测公平性'
        }
        
        for metric_key, metric_name in metrics_info.items():
            if metric_key in attr_report:
                metric_data = attr_report[metric_key]
                max_diff = metric_data.get('max_difference', 
                                         metric_data.get('max_tpr_difference', 0) or 
                                         metric_data.get('max_fpr_difference', 0))
                bias_detected = attr_report['bias_detected'].get(f'{metric_key}_bias', False)
                
                status = "⚠️  存在偏见" if bias_detected else "✅ 公平"
                print(f"    {metric_name}: 最大差异 {max_diff:.4f} ({status})")
    
    if overall_bias_found:
        print("\n🚨 检测到模型中存在潜在的公平性偏见，请进一步分析并采取措施改进模型。")
    else:
        print("\n✅ 模型在检测的属性上未发现明显的公平性偏见。")

def main():
    """
    主函数
    """
    print("开始执行模型公平性检测...")
    
    success = perform_fairness_analysis()
    
    if success:
        print("\n🎉 公平性检测完成!")
        print("详细报告已保存到 output/fairness_report.csv")
    else:
        print("\n💥 公平性检测失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()