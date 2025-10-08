import re

# 读取文件
with open('convert_train_data.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复未闭合的正则表达式
# 1. 修复大陆身份证号模式
content = re.sub(r"re\.match\(r'\^\\d\{17\}\[\\dXx\]", "re.match(r'^\d{17}[\dXx]$', value_str):")

# 2. 修复香港身份证号模式
content = re.sub(r"re\.match\(r'\^\[A-Z\]\{1,2\}\\d\{6\}\(\\\[0-9A\]\)", "re.match(r'^[A-Z]{1,2}\d{6}\([0-9A]\)$', value_str.upper()):")

# 3. 修复大陆手机号模式
content = re.sub(r"re\.match\(r'\^1\\\\d\{10\}", "re.match(r'^1\d{10}$', value_str):")

# 4. 修复邮箱模式
content = re.sub(r"re\.match\(r'\^\[a-zA-Z0-9\._%\+\-\]\+@\[a-zA-Z0-9\.-\]\+\\\.\\\[a-zA-Z\]\{2,\}", "re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value_str):")

# 写入修复后的内容
with open('convert_train_data_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("修复完成，已保存为 convert_train_data_fixed.py")