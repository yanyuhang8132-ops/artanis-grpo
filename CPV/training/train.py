import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import NetworkDataset, get_dataloader
from models.multimodal_model import MultimodalMultitaskModel
import matplotlib.pyplot as plt  # 导入matplotlib库

# 超参数设置
hidden_dim = 16  # 隐藏层维度
num_heads = 4  # 注意力头数
batch_size = 32  # 批大小
learning_rate = 0.001  # 学习率
num_epochs = 400  # 训练轮数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备

# 数据加载
train_json_path = "data/100_augmented_dataset_all.json"  # 训练集路径
# val_json_path = "data/val_dataset.json"  # 验证集路径

train_dataloader = get_dataloader(train_json_path, batch_size=batch_size, shuffle=True)
# val_dataloader = get_dataloader(val_json_path, batch_size=batch_size, shuffle=False)

# 从数据集中获取 N_max 和 interface_dim
N_max = train_dataloader.dataset.N_max
# interface_dim = len(train_dataloader.dataset.scenarios[0]["non_graph_features"]["interfaces"][0])

# 模型初始化
# model = MultimodalMultitaskModel(N_max=N_max, interface_dim=interface_dim, 
#                                   hidden_dim=hidden_dim, num_heads=num_heads)
# model.to(device)

flow_input_dim = train_dataloader.dataset[0]["non_graph"]["traffic_flows"].shape[1]
model = MultimodalMultitaskModel(
    N_max=N_max,
    flow_input_dim=flow_input_dim,
    ec_vocab_size=10000,
    ec_embed_dim=8,
    hidden_dim=hidden_dim,
    num_heads=num_heads
).to(device)


# 损失函数
criterion = nn.BCELoss()  # 二分类交叉熵损失

# 优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
losses = []  # 用于存储每个epoch的损失
for epoch in range(num_epochs):
    model.train()  # 训练模式
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        # 初始化批损失张量
        batch_loss = 0.0
        
        for sample in batch:
            # 数据移动到设备
            graph_data = {k: v.to(device) for k, v in sample["graph"].items()}
            non_graph_data = {k: v.to(device) for k, v in sample["non_graph"].items()}
            labels = {k: v.to(device) for k, v in sample["labels"].items()}
            mask = sample["mask"].to(device)
            
            # 前向传播
            outputs = model(graph_data, non_graph_data)
            
            # 初始化样本损失张量
            sample_loss = torch.tensor(0.0, device=device)
            
            # 计算各任务损失
            for task in outputs.keys():
                if task in ["high_util", "load_balancing"]:  # 忽略掉这两个
                    continue

                if task == "sla_labels":
                    # 不对其做 mask
                    valid_output = outputs[task]
                    valid_label = labels[task]
                elif task in ["reachability", "link_overload_pairs"]:
                    # 2D矩阵，使用二维mask
                    valid_output = outputs[task][mask][:, mask]
                    valid_label = labels[task][mask][:, mask]
                else:
                    # 1D向量型任务，使用一维mask
                    valid_output = outputs[task][mask]
                    valid_label = labels[task][mask]

                
                # 累加各任务损失
                sample_loss = sample_loss + criterion(valid_output, valid_label)
            
            # 累加样本损失到批损失
            batch_loss = batch_loss + sample_loss
        
        # 计算平均损失
        batch_loss = batch_loss / len(batch)
        
        # 反向传播
        batch_loss.backward()
        optimizer.step()
        
        # 记录损失
        total_loss += batch_loss.item()
        
        # 打印批次信息
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], "
                  f"Avg Loss: {avg_loss:.4f}")
    
    # 打印epoch信息
    avg_epoch_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")
    losses.append(avg_epoch_loss)  # 记录每个epoch的平均损失

# 绘制损失曲线
plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('training_loss_curve.png')  # 保存损失曲线图
plt.show()

# 保存模型
torch.save(model.state_dict(), "saved_models/multimodal_multitask_model_100.pth")
print("Training complete. Model saved.")