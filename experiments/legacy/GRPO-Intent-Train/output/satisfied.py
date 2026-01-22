import pandas as pd
import ast

def calculate_satisfaction_rate(csv_file):
    """
    计算满足率，排除min_bw_top_paths为空列表的样例
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 移除summary行
    df = df[df['sample_id'] != 'summary'].copy()
    
    # 处理min_bw_top_paths列，将字符串转换为列表
    def parse_list(x):
        if pd.isna(x) or x == '[]':
            return []
        try:
            return ast.literal_eval(x)
        except:
            return []
    
    df['min_bw_top_paths_parsed'] = df['min_bw_top_paths'].apply(parse_list)
    
    # 过滤掉min_bw_top_paths为空列表的样例
    df_filtered = df[df['min_bw_top_paths_parsed'].apply(lambda x: len(x) > 0)]
    
    # 计算满足率
    total_count = len(df_filtered)
    satisfied_count = len(df_filtered[df_filtered['satisfied'] == "True"])
    
    if total_count > 0:
        satisfaction_rate = (satisfied_count / total_count) * 100
    else:
        satisfaction_rate = 0
    
    return satisfaction_rate, satisfied_count, total_count

def main():
    # 文件路径
    file1 = "/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train/output/sla_check_results.csv"
    file2 = "/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train/output/sla_check_results_modified.csv"
    
    print("=" * 60)
    print("SLA满足率计算结果")
    print("=" * 60)
    
    # 计算第一个文件的满足率
    try:
        rate1, satisfied1, total1 = calculate_satisfaction_rate(file1)
        print(f"\n原始结果 (sla_check_results.csv):")
        print(f"  总样例数（排除空列表）: {total1}")
        print(f"  满足SLA的样例数: {satisfied1}")
        print(f"  满足率: {rate1:.2f}%")
    except Exception as e:
        print(f"处理文件1时出错: {e}")
    
    # 计算第二个文件的满足率
    try:
        rate2, satisfied2, total2 = calculate_satisfaction_rate(file2)
        print(f"\n修改后结果 (sla_check_results_modified.csv):")
        print(f"  总样例数: {total2}")
        print(f"  满足SLA的样例数: {satisfied2}")
        print(f"  满足率: {rate2:.2f}%")
    except Exception as e:
        print(f"处理文件2时出错: {e}")
    
    # 计算改进幅度
    if 'rate1' in locals() and 'rate2' in locals():
        improvement = rate2 - rate1
        print(f"\n改进情况:")
        print(f"  满足率提升: {improvement:.2f} 百分点")
        if rate1 > 0:
            relative_improvement = (improvement / rate1) * 100
            print(f"  相对提升: {relative_improvement:.2f}%")
    
    print("=" * 60)

if __name__ == "__main__":
    main()