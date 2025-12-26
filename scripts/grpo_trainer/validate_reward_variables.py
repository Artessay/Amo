#!/usr/bin/env python3
import os
import re
import ast
import sys

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def validate_file(file_path):
    """Validate a single file"""
    filename = os.path.basename(file_path)
    print(f"\nChecking file: {filename}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 提取变量值
    def extract_variable(var_name):
        match = re.search(rf"^{var_name}=(.*)$", content, re.MULTILINE)
        if match:
            value = match.group(1).strip()
            # 去除可能的引号包围
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            return value
        return None
    
    reward_function_path = extract_variable('REWARD_FUNCTION_PATH')
    reward_weights = extract_variable('REWARD_WEIGHTS')
    hv_reference_point = extract_variable('HV_REFERENCE_POINT')
    
    # 解析列表长度
    def get_list_length(value):
        if not value:
            return 0
        
        try:
            # 处理Python列表格式（带单引号）
            if value.startswith("['"):
                return len(ast.literal_eval(value))
            # 处理JSON数组格式（带双引号或无引号）
            elif value.startswith("["):
                return len(ast.literal_eval(value))
            # 单个值情况
            else:
                return 1
        except Exception as e:
            print(f"  Error parsing {value}: {e}")
            return -1
    
    # 计算各变量长度
    rf_len = get_list_length(reward_function_path)
    rw_len = get_list_length(reward_weights)
    hv_len = get_list_length(hv_reference_point)
    
    print(f"  REWARD_FUNCTION_PATH: {reward_function_path} -> length: {rf_len}")
    print(f"  REWARD_WEIGHTS: {reward_weights} -> length: {rw_len}")
    print(f"  HV_REFERENCE_POINT: {hv_reference_point} -> length: {hv_len}")
    
    # 检查长度一致性
    has_error = False
    
    # 如果定义了REWARD_FUNCTION_PATH为列表，检查其他变量
    if rf_len > 1:
        if rw_len > 0 and rw_len != rf_len:
            print(f"  ERROR: REWARD_WEIGHTS length ({rw_len}) does not match REWARD_FUNCTION_PATH length ({rf_len})")
            has_error = True
        if hv_len > 0 and hv_len != rf_len:
            print(f"  ERROR: HV_REFERENCE_POINT length ({hv_len}) does not match REWARD_FUNCTION_PATH length ({rf_len})")
            has_error = True
    
    # 检查REWARD_WEIGHTS和HV_REFERENCE_POINT之间的一致性
    if rw_len > 0 and hv_len > 0 and rw_len != hv_len:
        print(f"  ERROR: REWARD_WEIGHTS length ({rw_len}) does not match HV_REFERENCE_POINT length ({hv_len})")
        has_error = True
    
    if has_error:
        print(f"  ✗ Validation failed for {filename}!")
        return False
    else:
        print(f"  ✓ All reward variables have consistent lengths")
        return True

def validate_reward_variables():
    """Validate all .sh files in the directory"""
    all_valid = True
    
    for filename in os.listdir(SCRIPT_DIR):
        if not filename.endswith('.sh') or filename == os.path.basename(__file__):
            continue
        
        file_path = os.path.join(SCRIPT_DIR, filename)
        if not validate_file(file_path):
            all_valid = False
    
    return all_valid

if __name__ == "__main__":
    # Check if a specific file is requested
    if len(sys.argv) > 2 and sys.argv[1] == "--check":
        # Validate only the specified file
        if validate_file(sys.argv[2]):
            print(f"\n✓ {os.path.basename(sys.argv[2])} passed validation!")
            exit(0)
        else:
            print(f"\n✗ {os.path.basename(sys.argv[2])} failed validation!")
            exit(1)
    else:
        # Validate all files
        if validate_reward_variables():
            print("\n✓ All files passed validation!")
            exit(0)
        else:
            print("\n✗ Some files failed validation!")
            exit(1)
