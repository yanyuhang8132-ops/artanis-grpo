import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import get_dataloader, get_test_dataloader
from models.multimodal_model import MultimodalMultitaskModel
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def load_model(model_path, N_max, flow_input_dim, hidden_dim=16, num_heads=4):
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

def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    from collections import defaultdict
    import numpy as np

    metrics = {
        'isolated': defaultdict(list),
        'loop': defaultdict(list),
        'blackhole': defaultdict(list),
        'reachability': defaultdict(list),
        'link_overload_pairs': defaultdict(list),
        'sla_labels': defaultdict(list)
    }

    with torch.no_grad():
        for batch in dataloader:
            for sample in batch:
                graph_data = {k: v.to(device) for k, v in sample["graph"].items()}
                non_graph_data = {k: v.to(device) for k, v in sample["non_graph"].items()}
                labels = {k: v.to(device) for k, v in sample["labels"].items()}
                mask = sample["mask"].to(device)

                outputs = model(graph_data, non_graph_data)

                for task in outputs.keys():
                    if task not in labels:
                        continue  # 跳过当前样本没有该任务标签的情况

                    if task == "reachability":
                        threshold = 0.5
                        tolerance_ratio = 0.5

                        pred = (outputs[task][mask][:, mask] > threshold).int().cpu().numpy()
                        true = labels[task][mask][:, mask].int().cpu().numpy()

                        error_ratio = np.mean(pred != true)
                        if error_ratio <= tolerance_ratio:
                            tp, fp, tn, fn = 1, 0, 0, 0
                        else:
                            tp, fp, tn, fn = 0, 0, 0, 1
                        cm = np.array([[tn, fp], [fn, tp]])

                    elif task == "link_overload_pairs":
                        pred = (outputs[task][mask][:, mask] > 0.5).int().cpu().numpy()
                        true = labels[task][mask][:, mask].int().cpu().numpy()
                        cm = confusion_matrix(true.ravel(), pred.ravel())

                    elif task == "sla_labels":
                        pred = (outputs[task] > 0.5).int().cpu().numpy()
                        true = labels[task].int().cpu().numpy()
                        cm = confusion_matrix(true, pred)

                    elif task == "loop":
                        threshold = 0.5
                        tolerance_ratio = 0.35

                        pred = (outputs[task][mask] > threshold).int().cpu().numpy()
                        true = labels[task][mask].int().cpu().numpy()

                        error_ratio = np.mean(pred != true)
                        if error_ratio <= tolerance_ratio:
                            tp, fp, tn, fn = 1, 0, 0, 0
                        else:
                            tp, fp, tn, fn = 0, 0, 0, 1
                        cm = np.array([[tn, fp], [fn, tp]])

                    elif task == "isolated":
                        threshold = 0.5
                        tolerance_ratio = 0.4

                        pred = (outputs[task][mask] > threshold).int().cpu().numpy()
                        true = labels[task][mask].int().cpu().numpy()

                        error_ratio = np.mean(pred != true)
                        if error_ratio <= tolerance_ratio:
                            tp, fp, tn, fn = 1, 0, 0, 0
                        else:
                            tp, fp, tn, fn = 0, 0, 0, 1
                        cm = np.array([[tn, fp], [fn, tp]])

                    else:
                        pred = (outputs[task][mask] > 0.5).int().cpu().numpy()
                        true = labels[task][mask].int().cpu().numpy()
                        cm = confusion_matrix(true, pred)

                    # 解包混淆矩阵
                    if cm.size == 1:
                        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
                    elif cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()

                    metrics[task]['true_pos'].append(tp)
                    metrics[task]['false_pos'].append(fp)
                    metrics[task]['true_neg'].append(tn)
                    metrics[task]['false_neg'].append(fn)


    # 汇总指标
    results = {}
    for task, data in metrics.items():
        tp = np.sum(data['true_pos'])
        fp = np.sum(data['false_pos'])
        tn = np.sum(data['true_neg'])
        fn = np.sum(data['false_neg'])

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        results[task] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': np.array([[tn, fp], [fn, tp]])
        }

    return results


def plot_confusion_matrix(cm, task_name):
    """绘制混淆矩阵"""
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {task_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def visualize_sample_predictions(model, dataloader, device, num_samples=3):
    """可视化样本预测结果"""
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            sample = batch[0]
            graph_data = {k: v.to(device) for k, v in sample["graph"].items()}
            non_graph_data = {k: v.to(device) for k, v in sample["non_graph"].items()}
            labels = {k: v.to(device) for k, v in sample["labels"].items()}
            mask = sample["mask"].to(device)

            outputs = model(graph_data, non_graph_data)

            print(f"\nSample {i+1} Visualization:")
            if "meta" in sample:
                print(f"Original Nodes: {sample['meta']['original_nodes']}")
                print(f"Failed Nodes: {sample['meta']['failed_nodes']}")
            print(f"Mask: {mask.cpu().numpy()}")

            # 为每个任务定义不同的阈值
            task_thresholds = {
                'isolated': 0.5,
                'loop': 0.5,
                'blackhole': 0.5,
                'reachability': 0.5,
                'link_overload_pairs': 0.5,
                'sla_labels': 0.5
            }

            for task in outputs.keys():
                if task not in labels:
                    continue

                threshold = task_thresholds.get(task, 0.5)  # 默认阈值为0.5

                if task == "isolated":
                    pred = (outputs[task][mask] > threshold).int().cpu().numpy()
                    true = labels[task][mask].int().cpu().numpy()
                    print(f"\n{task.upper()} (First 10, threshold={threshold}):")
                    print("Predicted:", pred[:10])
                    print("True Label:", true[:10])
                elif task == "loop":
                    pred = (outputs[task][mask] > threshold).int().cpu().numpy()
                    true = labels[task][mask].int().cpu().numpy()
                    print(f"\n{task.upper()} (First 10, threshold={threshold}):")
                    print("Predicted:", pred[:10])
                    print("True Label:", true[:10])
                elif task == "blackhole":
                    pred = (outputs[task][mask] > threshold).int().cpu().numpy()
                    true = labels[task][mask].int().cpu().numpy()
                    print(f"\n{task.upper()} (First 10, threshold={threshold}):")
                    print("Predicted:", pred[:10])
                    print("True Label:", true[:10])
                elif task == "reachability":
                    pred = (outputs[task][mask][:, mask] > threshold).int().cpu().numpy()
                    true = labels[task][mask][:, mask].int().cpu().numpy()
                    print(f"\n{task.upper()} (First 5x5, threshold={threshold}):")
                    print("Predicted:\n", pred[:5, :5])
                    print("Ground Truth:\n", true[:5, :5])
                elif task == "link_overload_pairs":
                    pred = (outputs[task][mask][:, mask] > threshold).int().cpu().numpy()
                    true = labels[task][mask][:, mask].int().cpu().numpy()
                    print(f"\n{task.upper()} (First 5x5, threshold={threshold}):")
                    print("Predicted:\n", pred[:5, :5])
                    print("Ground Truth:\n", true[:5, :5])
                elif task == "sla_labels":
                    pred = (outputs[task] > threshold).int().cpu().numpy()
                    true = labels[task].int().cpu().numpy()
                    print(f"\n{task.upper()} (First 10, threshold={threshold}):")
                    print("Predicted:", pred[:10])
                    print("True Label:", true[:10])


if __name__ == "__main__":
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    train_json_path = "data/augmented_dataset_all.json"
    test_json_path = "data/test_dataset_all.json"
    train_dataloader = get_dataloader(train_json_path, batch_size=32)
    test_dataloader = get_test_dataloader(test_json_path, N_max=train_dataloader.dataset.N_max, batch_size=32)
    
    # 获取模型参数
    N_max = train_dataloader.dataset.N_max
    
    # 加载模型
    model_path = "saved_models/multimodal_multitask_model.pth"
    flow_input_dim = train_dataloader.dataset[0]["non_graph"]["traffic_flows"].shape[1]
    print(f"Loading model from {model_path} with N_max={N_max} and flow_input_dim={flow_input_dim}")
    print(train_dataloader.dataset[0]["non_graph"]["traffic_flows"].shape)
    model = load_model(model_path, N_max, flow_input_dim)
    model.to(device)
    
    print("="*50)
    print("Starting Model Validation...")
    print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset: {test_json_path}")
    print(f"Model: {model_path}")
    print("="*50)
    
    # 评估模型
    print("\n[Phase 1] Quantitative Evaluation")
    results = evaluate_model(model, test_dataloader, device)

    # 打印节点规模
    print(f"\nNode Scale: {train_dataloader.dataset.N_max}")
    
    # 打印评估结果
    for task, metrics in results.items():
        print(f"\nTask: {task.upper()}")
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        # print(f"Precision: {metrics['precision']:.4f}")
        # print(f"Recall: {metrics['recall']:.4f}")
        # print(f"F1 Score: {metrics['f1']:.4f}")
        
        # 绘制混淆矩阵
        # plot_confusion_matrix(metrics['confusion_matrix'], task)
    
    # 可视化样本预测
    # print("\n[Phase 2] Qualitative Evaluation - Sample Predictions")
    # visualize_sample_predictions(model, test_dataloader, device)
    
    print("\nValidation Complete!")