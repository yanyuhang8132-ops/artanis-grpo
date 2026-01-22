
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-whitegrid')

# 数据
scales = [14, 35, 70, 100]
routing_metrics = {
    'ISOLATED': [0.96, 0.94, 0.93, 0.94],
    'LOOP': [0.99, 0.99, 0.96, 0.96],
    'BLACKHOLE': [0.98, 0.98, 0.97, 0.96],
    'REACHABILITY': [1.00, 0.96, 0.94, 0.96]
}
traffic_metrics = {
    'LINK OVERLOAD': [0.96, 0.96, 0.94, 0.94],
    'SLA LABELS': [0.98, 0.93, 0.92, 0.93]
}
time_cost = [2.1, 33.7, 56.5, 64.6]

bar_width = 0.1
x = np.arange(len(scales))

fig, ax1 = plt.subplots(figsize=(10, 6))

# 路由级任务柱状图
for idx, (label, values) in enumerate(routing_metrics.items()):
    ax1.bar(x + idx * bar_width, values, width=bar_width, label=label)

# 流量级任务柱状图
for idx, (label, values) in enumerate(traffic_metrics.items()):
    ax1.bar(x + (idx + len(routing_metrics)) * bar_width, values, width=bar_width, label=label)

ax1.set_ylabel('验证准确率')
ax1.set_xlabel('网络规模（节点数）')
ax1.set_xticks(x + 2.5 * bar_width)
ax1.set_xticklabels(scales)
ax1.set_ylim(0, 1.1)

# 验证时间折线图
ax2 = ax1.twinx()
ax2.plot(x + 2.5 * bar_width, time_cost, color='gold', marker='o', linewidth=2.5, label='验证时间')
ax2.set_ylabel('平均验证时间 (ms)')
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True)

plt.title('不同任务验证效果与时间对比')
plt.tight_layout()
plt.savefig("validator_performance_plot.pdf")
plt.show()
