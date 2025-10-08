#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公平性和偏见检测工具
用于检测机器学习模型中的公平性和偏见问题
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class FairnessChecker:
    def __init__(self, data_df, label_col='Label', prediction_col='PredictedProb'):
        """
        初始化公平性检测器
        
        参数:
        data_df: 包含标签和预测结果的数据框
        label_col: 真实标签列名
        prediction_col: 预测概率列名
        """
        self.data_df = data_df
        self.label_col = label_col
        self.prediction_col = prediction_col
        
    def _get_metrics(self, subset_data, threshold=0.5):
        """
        计算子集的评估指标
        
        参数:
        subset_data: 数据子集
        threshold: 分类阈值
        
        返回:
        dict: 包含各种评估指标的字典
        """
        if len(subset_data) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'positive_rate': 0.0,
                'sample_count': 0
            }
            
        y_true = subset_data[self.label_col]
        y_pred = (subset_data[self.prediction_col] >= threshold).astype(int)
        
        # 处理全为0或全为1的情况
        if len(np.unique(y_true)) < 2:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        
        positive_rate = y_pred.mean()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'positive_rate': positive_rate,
            'sample_count': len(subset_data)
        }
    
    def demographic_parity(self, protected_attr, threshold=0.5):
        """
        计算不同受保护群体的预测正例率差异（Demographic Parity）
        
        参数:
        protected_attr: 受保护属性列名
        threshold: 分类阈值
        
        返回:
        dict: 包含各群体正例率和差异的字典
        """
        groups = self.data_df[protected_attr].unique()
        results = {}
        
        for group in groups:
            subset = self.data_df[self.data_df[protected_attr] == group]
            metrics = self._get_metrics(subset, threshold)
            results[group] = {
                'positive_rate': metrics['positive_rate'],
                'sample_count': metrics['sample_count']
            }
        
        # 计算最大差异
        positive_rates = [results[group]['positive_rate'] for group in groups]
        max_diff = max(positive_rates) - min(positive_rates) if positive_rates else 0
        
        return {
            'group_results': results,
            'max_difference': max_diff,
            'metric': 'positive_rate'
        }
    
    def equal_opportunity(self, protected_attr, threshold=0.5):
        """
        计算不同受保护群体的真正例率差异（Equal Opportunity）
        
        参数:
        protected_attr: 受保护属性列名
        threshold: 分类阈值
        
        返回:
        dict: 包含各群体真正例率和差异的字典
        """
        groups = self.data_df[protected_attr].unique()
        results = {}
        
        # 只考虑正例（标签为1）的样本
        positive_samples = self.data_df[self.data_df[self.label_col] == 1]
        
        for group in groups:
            subset = positive_samples[positive_samples[protected_attr] == group]
            metrics = self._get_metrics(subset, threshold)
            results[group] = {
                'true_positive_rate': metrics['recall'],  # recall即为真正例率
                'sample_count': metrics['sample_count']
            }
        
        # 计算最大差异
        tpr_rates = [results[group]['true_positive_rate'] for group in groups]
        max_diff = max(tpr_rates) - min(tpr_rates) if tpr_rates else 0
        
        return {
            'group_results': results,
            'max_difference': max_diff,
            'metric': 'true_positive_rate'
        }
    
    def equalized_odds(self, protected_attr, threshold=0.5):
        """
        计算不同受保护群体的真正例率和假正例率差异（Equalized Odds）
        
        参数:
        protected_attr: 受保护属性列名
        threshold: 分类阈值
        
        返回:
        dict: 包含各群体真正例率、假正例率和差异的字典
        """
        groups = self.data_df[protected_attr].unique()
        results = {}
        
        # 正例样本（标签为1）
        positive_samples = self.data_df[self.data_df[self.label_col] == 1]
        # 负例样本（标签为0）
        negative_samples = self.data_df[self.data_df[self.label_col] == 0]
        
        for group in groups:
            # 正例群体
            pos_subset = positive_samples[positive_samples[protected_attr] == group]
            pos_metrics = self._get_metrics(pos_subset, threshold)
            
            # 负例群体
            neg_subset = negative_samples[negative_samples[protected_attr] == group]
            neg_metrics = self._get_metrics(neg_subset, threshold)
            
            results[group] = {
                'true_positive_rate': pos_metrics['recall'],
                'false_positive_rate': neg_metrics['positive_rate'],  # 假正例率等于负例中被预测为正例的比例
                'positive_sample_count': pos_metrics['sample_count'],
                'negative_sample_count': neg_metrics['sample_count']
            }
        
        # 计算最大差异
        tpr_rates = [results[group]['true_positive_rate'] for group in groups]
        fpr_rates = [results[group]['false_positive_rate'] for group in groups]
        max_tpr_diff = max(tpr_rates) - min(tpr_rates) if tpr_rates else 0
        max_fpr_diff = max(fpr_rates) - min(fpr_rates) if fpr_rates else 0
        
        return {
            'group_results': results,
            'max_tpr_difference': max_tpr_diff,
            'max_fpr_difference': max_fpr_diff,
            'metric': 'true_positive_rate_and_false_positive_rate'
        }
    
    def predictive_parity(self, protected_attr, threshold=0.5):
        """
        计算不同受保护群体的正预测值差异（Predictive Parity）
        
        参数:
        protected_attr: 受保护属性列名
        threshold: 分类阈值
        
        返回:
        dict: 包含各群体正预测值和差异的字典
        """
        groups = self.data_df[protected_attr].unique()
        results = {}
        
        for group in groups:
            subset = self.data_df[self.data_df[protected_attr] == group]
            metrics = self._get_metrics(subset, threshold)
            results[group] = {
                'positive_predictive_value': metrics['precision'],
                'sample_count': metrics['sample_count']
            }
        
        # 计算最大差异
        ppv_rates = [results[group]['positive_predictive_value'] for group in groups]
        max_diff = max(ppv_rates) - min(ppv_rates) if ppv_rates else 0
        
        return {
            'group_results': results,
            'max_difference': max_diff,
            'metric': 'positive_predictive_value'
        }
    
    def check_bias(self, protected_attr, threshold=0.5, fairness_threshold=0.1):
        """
        综合检查模型的公平性偏见
        
        参数:
        protected_attr: 受保护属性列名
        threshold: 分类阈值
        fairness_threshold: 公平性差异阈值，超过此值认为存在偏见
        
        返回:
        dict: 包含所有公平性指标检查结果的字典
        """
        # 检查各个公平性指标
        dp_result = self.demographic_parity(protected_attr, threshold)
        eo_result = self.equal_opportunity(protected_attr, threshold)
        eo_odds_result = self.equalized_odds(protected_attr, threshold)
        pp_result = self.predictive_parity(protected_attr, threshold)
        
        # 判断是否存在偏见
        bias_detected = {
            'demographic_parity_bias': dp_result['max_difference'] > fairness_threshold,
            'equal_opportunity_bias': eo_result['max_difference'] > fairness_threshold,
            'equalized_odds_bias': (
                eo_odds_result['max_tpr_difference'] > fairness_threshold or 
                eo_odds_result['max_fpr_difference'] > fairness_threshold
            ),
            'predictive_parity_bias': pp_result['max_difference'] > fairness_threshold
        }
        
        return {
            'demographic_parity': dp_result,
            'equal_opportunity': eo_result,
            'equalized_odds': eo_odds_result,
            'predictive_parity': pp_result,
            'bias_detected': bias_detected,
            'overall_bias': any(bias_detected.values()),
            'protected_attribute': protected_attr
        }
    
    def generate_fairness_report(self, protected_attrs, threshold=0.5, fairness_threshold=0.1):
        """
        生成完整的公平性检测报告
        
        参数:
        protected_attrs: 受保护属性列名列表
        threshold: 分类阈值
        fairness_threshold: 公平性差异阈值
        
        返回:
        dict: 包含所有受保护属性公平性检测结果的报告
        """
        report = {}
        
        for attr in protected_attrs:
            if attr in self.data_df.columns:
                print(f"正在检查属性 '{attr}' 的公平性...")
                report[attr] = self.check_bias(attr, threshold, fairness_threshold)
            else:
                print(f"警告: 属性 '{attr}' 在数据中不存在")
                report[attr] = {
                    'error': f"属性 '{attr}' 在数据中不存在"
                }
        
        return report

def identify_potential_sensitive_features(data_df):
    """
    识别数据中潜在的敏感特征
    
    参数:
    data_df: 数据框
    
    返回:
    list: 潜在敏感特征列表
    """
    # 敏感特征关键词（中英文）
    sensitive_keywords = [
        # 性别相关
        'gender', 'sex', '性别', '性別',
        # 年龄相关
        'age', '年龄', '年齡',
        # 种族/民族相关
        'race', 'ethnicity', '民族', '族群', '种族', '種族',
        # 地区相关
        'region', '地区', '地區', 'location', '地址', 'address',
        # 部门/网点相关
        'department', '部门', '网点', 'branch', 'belong',
        # 渠道相关
        'channel', '渠道'
    ]
    
    potential_sensitive_features = []
    
    for col in data_df.columns:
        col_lower = col.lower()
        for keyword in sensitive_keywords:
            if keyword in col_lower:
                potential_sensitive_features.append(col)
                break
    
    return potential_sensitive_features

# 使用示例
if __name__ == "__main__":
    # 示例用法
    print("公平性检测工具已准备就绪")
    print("请使用 FairnessChecker 类来检测模型的公平性")