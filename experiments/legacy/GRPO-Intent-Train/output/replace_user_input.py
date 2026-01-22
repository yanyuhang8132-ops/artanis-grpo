import json
import os

def fix_model_output(model_output):
    """
    检测并修复model_output字段，确保有外层中括号
    """
    if not model_output:
        return model_output
    
    model_output = model_output.strip()
    
    # 检查是否已经有外层中括号
    if model_output.startswith('[') and model_output.endswith(']'):
        return model_output
    
    # 如果没有外层中括号，添加中括号
    return f"[{model_output}]"

def replace_user_input():
    # 定义文件路径
    source_file = "2.jsonl"  # 源文件，包含要替换的user_input内容
    target_file = "4.jsonl"  # 目标文件，需要被替换user_input的文件
    output_file = "4_updated.jsonl"  # 输出文件
    
    # 检查文件是否存在
    if not os.path.exists(source_file):
        print(f"错误: 源文件 {source_file} 不存在")
        return
    
    if not os.path.exists(target_file):
        print(f"错误: 目标文件 {target_file} 不存在")
        return
    
    # 读取源文件中的user_input内容
    source_user_inputs = []
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'user_input' in data:
                            source_user_inputs.append(data['user_input'])
                        else:
                            print(f"警告: 第 {line_num} 行没有 'user_input' 字段")
                            source_user_inputs.append("")
                    except json.JSONDecodeError as e:
                        print(f"错误: 第 {line_num} 行JSON解析失败: {e}")
                        return
    except Exception as e:
        print(f"错误: 读取源文件失败: {e}")
        return
    
    # 读取目标文件并替换user_input字段
    updated_data = []
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # 检查是否有对应的源数据
                        if line_num - 1 < len(source_user_inputs):
                            # 替换user_input字段
                            data['user_input'] = source_user_inputs[line_num - 1]
                            
                        # 修复model_output字段，确保有外层中括号
                        if 'model_output' in data:
                            data['model_output'] = fix_model_output(data['model_output'])
                            
                        updated_data.append(data)
                        
                        if line_num - 1 >= len(source_user_inputs):
                            print(f"警告: 第 {line_num} 行没有对应的源数据")
                    except json.JSONDecodeError as e:
                        print(f"错误: 第 {line_num} 行JSON解析失败: {e}")
                        return
    except Exception as e:
        print(f"错误: 读取目标文件失败: {e}")
        return
    
    # 写入更新后的数据到输出文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for data in updated_data:
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"成功! 已将更新后的数据写入 {output_file}")
        print(f"共处理了 {len(updated_data)} 行数据")
        print(f"源文件包含 {len(source_user_inputs)} 个user_input")
        
    except Exception as e:
        print(f"错误: 写入输出文件失败: {e}")

def main():
    print("开始替换user_input字段...")
    replace_user_input()
    print("处理完成!")

if __name__ == "__main__":
    main()