import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import pandas as pd

# 为了在没有tkinter的环境中也能查看代码结构，我们使用条件导入
try:
    from convert_train_data import main as convert_train_data_main
    from add_train_label import add_label_to_wide_table
    from convert_predict_data import main as convert_predict_data_main
    from check_model_fairness import calculate_fairness_metrics
    TKINTER_AVAILABLE = True
    
    # 尝试导入可能依赖PyTorch的模块
    try:
        from train_model import gbdt_lr_train, preProcess
        from base.base_model_processor import BaseModelProcessor
        from predict import PredictModel
    except ImportError as e:
        print(f"Warning: Failed to import PyTorch-dependent modules: {e}")
        gbdt_lr_train = preProcess = BaseModelProcessor = PredictModel = None
except ImportError as e:
    print(f"Import error: {e}")
    TKINTER_AVAILABLE = False

class FinanceDataAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("金融数据处理与机器学习工具")
        self.root.geometry("800x600")
        
        # 设置窗口图标
        self.set_window_icon()
        
        # 显示当前工作目录
        current_dir_frame = ttk.Frame(root)
        current_dir_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)
        ttk.Label(current_dir_frame, text=f"当前工作目录: {os.getcwd()}").grid(row=0, column=0, sticky=tk.W)
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建标签页
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        # 创建各个功能标签页
        self.create_data_conversion_tab()
        self.create_label_addition_tab()
        self.create_model_training_tab()
        self.create_predict_data_conversion_tab()
        self.create_prediction_tab()
        self.create_fairness_check_tab()
        self.create_notes_tab()
        
        # 创建日志输出区域
        self.create_log_area()
        
    def create_data_conversion_tab(self):
        """创建训练数据转换标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="训练数据转换")
        
        # 文件选择
        ttk.Label(tab, text="训练数据目录:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.train_data_dir_var = tk.StringVar(value=os.path.join(os.getcwd(), "data_train"))
        ttk.Entry(tab, textvariable=self.train_data_dir_var, width=50).grid(row=0, column=1, pady=5)
        ttk.Button(tab, text="浏览", command=self.browse_train_data_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # 参数设置
        ttk.Label(tab, text="覆盖率阈值:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.coverage_threshold_var = tk.DoubleVar(value=0.95)
        ttk.Entry(tab, textvariable=self.coverage_threshold_var, width=20).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(tab, text="最大TopK:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.max_top_k_var = tk.IntVar(value=50)
        ttk.Entry(tab, textvariable=self.max_top_k_var, width=20).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 提示信息
        ttk.Label(tab, text="注意：执行数据转换前，请确保已完成配置文件的设置", foreground="red").grid(row=3, column=0, columnspan=3, pady=5)
        ttk.Label(tab, text="配置文件包括：primary_key.csv、category_type.csv、label_key.csv、sensitive_attr.csv", foreground="red").grid(row=4, column=0, columnspan=3, pady=5)
        
        # 执行按钮
        ttk.Button(tab, text="执行训练数据转换", command=self.run_data_conversion).grid(row=5, column=0, columnspan=3, pady=20)
        
    def create_label_addition_tab(self):
        """创建标签添加标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="标签添加")
        
        # 文件选择
        ttk.Label(tab, text="标签Excel文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.label_file_var = tk.StringVar(value=os.path.join(os.getcwd(), "label_train"))
        ttk.Entry(tab, textvariable=self.label_file_var, width=50).grid(row=0, column=1, pady=5)
        ttk.Button(tab, text="浏览", command=self.browse_label_file).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(tab, text="工作表名称:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.sheet_name_var = tk.StringVar(value="2024Raw")
        ttk.Entry(tab, textvariable=self.sheet_name_var, width=20).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(tab, text="标签列名:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.label_column_var = tk.StringVar(value="本地支薪")
        ttk.Entry(tab, textvariable=self.label_column_var, width=20).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 样本平衡参数
        ttk.Label(tab, text="样本平衡比例:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.balance_ratio_var = tk.StringVar(value="")
        ttk.Entry(tab, textvariable=self.balance_ratio_var, width=20).grid(row=3, column=1, sticky=tk.W, pady=5)
        ttk.Label(tab, text="（留空表示不进行平衡）").grid(row=3, column=2, sticky=tk.W, pady=5)
        
        # 参数说明
        balance_info_text = """
参数说明：
- 留空：不进行样本平衡处理
- 0.5：负样本:正样本 = 1:2
- 1：负样本:正样本 = 1:1
- 10：负样本:正样本 = 10:1
        """.strip()
        ttk.Label(tab, text=balance_info_text, foreground="blue").grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # 执行按钮
        ttk.Button(tab, text="添加标签", command=self.run_label_addition).grid(row=5, column=0, columnspan=3, pady=20)
        
    def create_model_training_tab(self):
        """创建模型训练标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="模型训练")
        
        # 执行按钮
        ttk.Button(tab, text="开始训练", command=self.run_model_training).grid(row=0, column=0, columnspan=2, pady=20)
        
    def create_prediction_tab(self):
        """创建预测标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="模型预测")
        
        # SHAP选项
        self.calculate_shap_var = tk.BooleanVar()
        ttk.Checkbutton(tab, text="计算SHAP值（较慢但提供可解释性分析）", variable=self.calculate_shap_var).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # 执行按钮
        ttk.Button(tab, text="执行预测", command=self.run_prediction).grid(row=1, column=0, columnspan=3, pady=20)
        
    def create_predict_data_conversion_tab(self):
        """创建预测数据转换标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="预测数据转换")
        
        # 文件选择
        ttk.Label(tab, text="预测数据目录:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.predict_data_dir_var = tk.StringVar(value=os.path.join(os.getcwd(), "data_predict"))
        ttk.Entry(tab, textvariable=self.predict_data_dir_var, width=50).grid(row=0, column=1, pady=5)
        ttk.Button(tab, text="浏览", command=self.browse_predict_data_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # 参数设置
        ttk.Label(tab, text="覆盖率阈值:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.predict_coverage_threshold_var = tk.DoubleVar(value=0.95)
        ttk.Entry(tab, textvariable=self.predict_coverage_threshold_var, width=20).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(tab, text="最大TopK:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.predict_max_top_k_var = tk.IntVar(value=50)
        ttk.Entry(tab, textvariable=self.predict_max_top_k_var, width=20).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 执行按钮
        ttk.Button(tab, text="执行预测数据转换", command=self.run_predict_data_conversion).grid(row=3, column=0, columnspan=3, pady=20)
        
    def create_fairness_check_tab(self):
        """创建公平性检测标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="公平性检测")
        
        # 执行按钮
        ttk.Button(tab, text="检测模型公平性", command=self.run_fairness_check).grid(row=0, column=0, columnspan=2, pady=20)
        
    def create_notes_tab(self):
        """创建注意事项标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="注意事项")
        
        # 创建一个文本框来显示注意事项
        notes_frame = ttk.LabelFrame(tab, text="重要提示", padding="10")
        notes_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        notes_frame.columnconfigure(0, weight=1)
        notes_frame.rowconfigure(0, weight=1)
        
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        
        # 注意事项内容
        notes_text = """
数据准备要求：
- 确保Excel文件格式正确，无损坏
- 文件名和列名应使用英文或数字，避免特殊字符
- 数值型数据应为纯数字格式，避免包含文本或符号

模型维护建议：
- 建议每季度使用最新数据重新训练模型
- 当业务环境发生重大变化时，应及时更新模型

推荐使用建议：
- 推荐结果仅供参考，应结合业务经验进行综合判断
- 对于高价值客户，建议优先跟进和营销

公平性检测建议：
- 定期执行公平性检测，确保模型不会对特定群体产生歧视
- 当公平性指标得分较低时，应分析原因并考虑重新训练模型
- 可根据不同敏感属性分别进行公平性检测

配置文件说明：
1. primary_key.csv：定义各Excel数据文件的主键字段
   - 文件名必须与data_train/目录下的文件名完全一致
   - 主键字段用于唯一标识每条记录

2. category_type.csv：指定需要强制作为类别特征处理的字段
   - 解决数值型字段但业务上应为类别型的问题

3. label_key.csv：定义标签文件的信息
   - 文件名必须与label_train/目录下的文件名完全一致
   - 标签列名包含目标变量

4. sensitive_attr.csv：定义用于公平性检测的敏感属性字段
   - 用于从指定文件中提取敏感属性进行公平性分析

参数设置说明：
- 覆盖率阈值：用于控制类别特征的覆盖率，如0.95表示保留覆盖95%数据的类别值
- 最大TopK：限制类别特征的最大取值数量，防止特征维度爆炸
- 当两个参数冲突时，系统优先遵循最大TopK限制

常见问题：
- 推荐结果不准确：请检查输入数据质量，确保数据完整且格式正确
- 如何解释高价值推荐：使用SHAP值计算功能获取详细解释报告
- 公平性检测得分较低：检查敏感属性选择、分析数据分布偏差、考虑重新训练模型
        """.strip()
        
        # 创建文本框显示注意事项
        text_widget = tk.Text(notes_frame, wrap=tk.WORD, height=20)
        text_widget.insert(tk.END, notes_text)
        text_widget.configure(state='disabled')  # 设置为只读
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(notes_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
    def create_log_area(self):
        """创建日志输出区域"""
        log_frame = ttk.LabelFrame(self.main_frame, text="运行日志", padding="10")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.main_frame.rowconfigure(1, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def set_window_icon(self):
        """设置窗口图标"""
        try:
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 构建图标文件路径
            icon_path = os.path.join(current_dir, "images", "ai.ico")
            # 设置窗口图标
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except Exception as e:
            # 如果设置图标失败，不中断程序运行
            print(f"设置窗口图标失败: {e}")
            
    def browse_train_data_dir(self):
        """浏览训练数据目录"""
        directory = filedialog.askdirectory()
        if directory:
            self.train_data_dir_var.set(directory)
            
    def browse_label_file(self):
        """浏览标签文件"""
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.label_file_var.set(file_path)
            
    def browse_predict_data_dir(self):
        """浏览预测数据目录"""
        directory = filedialog.askdirectory()
        if directory:
            self.predict_data_dir_var.set(directory)
            
    def log_message(self, message):
        """在日志区域显示消息"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def run_data_conversion(self):
        """执行数据转换"""
        def task():
            try:
                self.log_message("开始数据转换...")
                # 调用实际的数据转换函数
                convert_train_data_main(
                    coverage_threshold=self.coverage_threshold_var.get(),
                    max_top_k=self.max_top_k_var.get()
                )
                self.log_message("✅ 数据转换完成!")
            except Exception as e:
                self.log_message(f"❌ 数据转换失败: {str(e)}")
                
        threading.Thread(target=task, daemon=True).start()
        
    def run_label_addition(self):
        """执行标签添加"""
        def task():
            try:
                self.log_message("开始添加标签...")
                # 获取平衡比例参数
                balance_ratio_str = self.balance_ratio_var.get()
                balance_ratio = None
                if balance_ratio_str:
                    try:
                        balance_ratio = float(balance_ratio_str)
                    except ValueError:
                        self.log_message("⚠️  样本平衡比例参数无效，将不进行样本平衡处理")
                
                # 调用实际的标签添加函数
                add_label_to_wide_table(
                    excel_file_path=self.label_file_var.get(),
                    sheet_name=self.sheet_name_var.get(),
                    label_column=self.label_column_var.get(),
                    balance_ratio=balance_ratio
                )
                if balance_ratio is not None:
                    self.log_message(f"✅ 标签添加完成! (已按 {balance_ratio}:1 的比例进行样本平衡)")
                else:
                    self.log_message("✅ 标签添加完成! (未进行样本平衡处理)")
            except Exception as e:
                self.log_message(f"❌ 标签添加失败: {str(e)}")
                
        threading.Thread(target=task, daemon=True).start()
        
    def run_model_training(self):
        """执行模型训练"""
        def task():
            try:
                # 检查必要的模块是否可用
                if preProcess is None or gbdt_lr_train is None or BaseModelProcessor is None:
                    self.log_message("❌ 模型训练功能不可用: 未安装PyTorch或相关依赖缺失")
                    return
                    
                self.log_message("开始数据预处理...")
                data = preProcess()
                
                self.log_message("加载特征配置...")
                feature_config = pd.read_csv('config/features.csv')
                continuous_feature = feature_config[feature_config['feature_type'] == 'continuous']['feature_name'].tolist()
                category_feature = feature_config[feature_config['feature_type'] == 'category']['feature_name'].tolist()
                
                # 显示大模型解读提示
                processor = BaseModelProcessor()
                processor.show_model_interpretation_prompt()
                
                self.log_message("开始训练GBDT+LR模型...")
                gbdt_lr_train(data, category_feature, continuous_feature)
                self.log_message("✅ 模型训练完成!")
            except Exception as e:
                self.log_message(f"❌ 模型训练失败: {str(e)}")
                
        threading.Thread(target=task, daemon=True).start()
        
    def run_prediction(self):
        """执行预测"""
        def task():
            try:
                # 检查必要的模块是否可用
                if PredictModel is None:
                    self.log_message("❌ 预测功能不可用: 未安装PyTorch或相关依赖缺失")
                    return
                    
                self.log_message("开始预测...")
                predictor = PredictModel()
                predict_data_path = "output/ml_wide_table_predict_global.csv"
                output_path = "output/prediction_results.csv"
                calculate_shap = self.calculate_shap_var.get()
                
                success = predictor.run_prediction(predict_data_path, output_path, calculate_shap)
                
                if success:
                    self.log_message("✅ 数据预测完成!")
                    if calculate_shap:
                        self.log_message("✅ 特征贡献值已计算并包含在结果中")
                    else:
                        self.log_message("ℹ️  仅进行预测，未计算SHAP值")
                else:
                    self.log_message("❌ 数据预测失败!")
            except Exception as e:
                self.log_message(f"❌ 数据预测失败: {str(e)}")
                
        threading.Thread(target=task, daemon=True).start()
        
    def run_predict_data_conversion(self):
        """执行预测数据转换"""
        def task():
            try:
                self.log_message("开始预测数据转换...")
                # 调用实际的预测数据转换函数
                convert_predict_data_main(
                    coverage_threshold=0.95,
                    max_top_k=50
                )
                self.log_message("✅ 预测数据转换完成!")
            except Exception as e:
                self.log_message(f"❌ 预测数据转换失败: {str(e)}")
                
        threading.Thread(target=task, daemon=True).start()
        
    def run_fairness_check(self):
        """执行公平性检测"""
        def task():
            try:
                self.log_message("开始检测模型公平性...")
                # 调用实际的公平性检测函数
                calculate_fairness_metrics()
                self.log_message("✅ 公平性检测完成!")
            except Exception as e:
                self.log_message(f"❌ 公平性检测失败: {str(e)}")
                
        threading.Thread(target=task, daemon=True).start()

def main():
    # 检查tkinter是否可用
    if not TKINTER_AVAILABLE:
        print("Tkinter不可用，请确保已安装Python Tkinter扩展")
        print("在Windows上，通常Python安装包已包含Tkinter")
        print("在Linux上，可能需要安装python3-tk或类似包")
        return
    
    root = tk.Tk()
    app = FinanceDataAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
