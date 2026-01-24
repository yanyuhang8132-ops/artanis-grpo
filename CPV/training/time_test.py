import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from dataloader import get_dataloader, get_test_dataloader
from models.multimodal_model import MultimodalMultitaskModel

def load_model(model_path, N_max, flow_input_dim, hidden_dim=16, num_heads=4):
    """加载模型"""
    model = MultimodalMultitaskModel(
        N_max=N_max,
        flow_input_dim=flow_input_dim,
        ec_vocab_size=10000,
        ec_embed_dim=8,
        hidden_dim=hidden_dim,
        num_heads=num_heads
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def test_inference_time(model, dataloader, device, warmup_batches=5):
    """测试模型推理时间"""
    total_samples = 0
    total_batches = 0
    batch_times = []
    
    print("开始预热...")
    # 预热阶段
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= warmup_batches:
                break
                
            for sample in batch:
                graph_data = {k: v.to(device) for k, v in sample["graph"].items()}
                non_graph_data = {k: v.to(device) for k, v in sample["non_graph"].items()}
                
                # 预热推理
                _ = model(graph_data, non_graph_data)
    
    print(f"预热完成，开始正式测试...")
    
    # 正式测试阶段
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            
            for sample in batch:
                graph_data = {k: v.to(device) for k, v in sample["graph"].items()}
                non_graph_data = {k: v.to(device) for k, v in sample["non_graph"].items()}
                
                # 单样本推理时间测试
                sample_start_time = time.time()
                outputs = model(graph_data, non_graph_data)
                sample_end_time = time.time()
                
                total_samples += 1
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            total_batches += 1
            
            # 每处理10个batch显示一次进度
            if (batch_idx + 1) % 10 == 0:
                avg_batch_time = np.mean(batch_times[-10:])
                print(f"已处理 {batch_idx + 1} 个batch, 最近10个batch平均时间: {avg_batch_time:.4f}s")
    
    return total_samples, total_batches, batch_times

def print_timing_statistics(total_samples, total_batches, batch_times):
    """打印时间统计信息"""
    total_time = sum(batch_times)
    avg_batch_time = np.mean(batch_times)
    avg_sample_time = total_time / total_samples
    
    print("=" * 60)
    print("时间测试结果统计:")
    print("=" * 60)
    print(f"总样本数: {total_samples}")
    print(f"总批次数: {total_batches}")
    print(f"总用时: {total_time:.4f} 秒")
    print(f"平均每批次用时: {avg_batch_time:.4f} 秒")
    print(f"平均每样本用时: {avg_sample_time:.4f} 秒")
    print(f"每秒处理样本数: {total_samples / total_time:.2f} 样本/秒")
    print("=" * 60)
    
    # 打印批次时间分布统计
    print("\n批次时间分布:")
    print(f"最小批次时间: {np.min(batch_times):.4f} 秒")
    print(f"最大批次时间: {np.max(batch_times):.4f} 秒")
    print(f"中位数批次时间: {np.median(batch_times):.4f} 秒")
    print(f"批次时间标准差: {np.std(batch_times):.4f} 秒")

if __name__ == "__main__":
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # 加载数据
    train_json_path = "data/100_augmented_dataset_all.json"
    test_json_path = "data/100_augmented_dataset_all.json"
    train_dataloader = get_dataloader(train_json_path, batch_size=32)
    test_dataloader = get_test_dataloader(test_json_path, N_max=train_dataloader.dataset.N_max, batch_size=32)
    
    # 获取模型参数
    N_max = train_dataloader.dataset.N_max
    flow_input_dim = train_dataloader.dataset[0]["non_graph"]["traffic_flows"].shape[1]
    
    # 加载模型
    model_path = "saved_models/multimodal_multitask_model_100.pth"
    print(f"加载模型: {model_path}")
    print(f"模型参数 - N_max: {N_max}, flow_input_dim: {flow_input_dim}")
    
    model = load_model(model_path, N_max, flow_input_dim)
    model.to(device)
    
    print("\n开始模型推理时间测试...")
    print(f"数据集: {test_json_path}")
    print(f"批次大小: {test_dataloader.batch_size}")
    
    # 开始时间测试
    start_time = time.time()
    total_samples, total_batches, batch_times = test_inference_time(model, test_dataloader, device)
    end_time = time.time()
    
    # 打印结果
    print_timing_statistics(total_samples, total_batches, batch_times)
    
    print(f"\n测试完成! 总耗时: {end_time - start_time:.4f} 秒")