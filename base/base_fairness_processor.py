import pandas as pd
import numpy as np

class BaseFairnessProcessor:
    """公平性检测处理器"""
    
    @staticmethod
    def calculate_demographic_parity(y_pred, sensitive_attr):
        """
        计算 Demographic Parity (人口统计学公平性)
        """
        groups = np.unique(sensitive_attr)
        positive_rates = []
        
        for group in groups:
            mask = sensitive_attr == group
            positive_rate = np.mean(y_pred[mask])
            positive_rates.append(positive_rate)
        
        # 计算最大差异作为不公平度量
        dp_diff = np.max(positive_rates) - np.min(positive_rates)
        return 1 - dp_diff  # 转换为公平性得分 (0-1之间，1表示完全公平)

    @staticmethod
    def calculate_equal_opportunity(y_true, y_pred, sensitive_attr):
        """
        计算 Equal Opportunity (机会均等)
        """
        groups = np.unique(sensitive_attr)
        tpr_rates = []
        
        for group in groups:
            mask = (sensitive_attr == group) & (y_true == 1)
            if np.sum(mask) > 0:
                tpr = np.mean(y_pred[mask])
            else:
                tpr = 0
            tpr_rates.append(tpr)
        
        # 计算最大差异作为不公平度量
        eo_diff = np.max(tpr_rates) - np.min(tpr_rates)
        return 1 - eo_diff  # 转换为公平性得分 (0-1之间，1表示完全公平)

    @staticmethod
    def calculate_equalized_odds(y_true, y_pred, sensitive_attr):
        """
        计算 Equalized Odds (均衡几率)
        """
        groups = np.unique(sensitive_attr)
        tpr_rates = []
        fpr_rates = []
        
        for group in groups:
            # 计算真阳性率 (TPR)
            tp_mask = (sensitive_attr == group) & (y_true == 1)
            if np.sum(tp_mask) > 0:
                tpr = np.mean(y_pred[tp_mask])
            else:
                tpr = 0
            tpr_rates.append(tpr)
            
            # 计算假阳性率 (FPR)
            fp_mask = (sensitive_attr == group) & (y_true == 0)
            if np.sum(fp_mask) > 0:
                fpr = np.mean(y_pred[fp_mask])
            else:
                fpr = 0
            fpr_rates.append(fpr)
        
        # 计算TPR和FPR的最大差异
        tpr_diff = np.max(tpr_rates) - np.min(tpr_rates)
        fpr_diff = np.max(fpr_rates) - np.min(fpr_rates)
        
        # 综合不公平度量
        eo_diff = (tpr_diff + fpr_diff) / 2
        return 1 - eo_diff  # 转换为公平性得分 (0-1之间，1表示完全公平)

    @staticmethod
    def calculate_predictive_parity(y_true, y_pred, sensitive_attr):
        """
        计算 Predictive Parity (预测公平性)
        """
        groups = np.unique(sensitive_attr)
        ppv_rates = []
        
        for group in groups:
            # 计算预测值为正的样本中真实值为正的比例 (Positive Predictive Value, PPV)
            pred_pos_mask = y_pred == 1
            group_mask = sensitive_attr == group
            combined_mask = pred_pos_mask & group_mask
            
            if np.sum(combined_mask) > 0:
                ppv = np.mean(y_true[combined_mask])
            else:
                ppv = 0
            ppv_rates.append(ppv)
        
        # 计算最大差异作为不公平度量
        pp_diff = np.max(ppv_rates) - np.min(ppv_rates)
        return 1 - pp_diff  # 转换为公平性得分 (0-1之间，1表示完全公平)

    @staticmethod
    def calculate_fairness_metrics(y_true, y_pred, sensitive_attr):
        """
        计算所有公平性指标
        """
        if sensitive_attr is None:
            return None
            
        # 将预测概率转换为二值预测（阈值设为0.5）
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        # 计算各种公平性指标
        demographic_parity = BaseFairnessProcessor.calculate_demographic_parity(y_pred_binary, sensitive_attr)
        equal_opportunity = BaseFairnessProcessor.calculate_equal_opportunity(y_true, y_pred_binary, sensitive_attr)
        equalized_odds = BaseFairnessProcessor.calculate_equalized_odds(y_true, y_pred_binary, sensitive_attr)
        predictive_parity = BaseFairnessProcessor.calculate_predictive_parity(y_true, y_pred_binary, sensitive_attr)
        
        # 返回结果
        fairness_metrics = pd.DataFrame({
            'Metric': ['Demographic Parity', 'Equal Opportunity', 'Equalized Odds', 'Predictive Parity'],
            'Score': [demographic_parity, equal_opportunity, equalized_odds, predictive_parity]
        })
        
        return fairness_metrics