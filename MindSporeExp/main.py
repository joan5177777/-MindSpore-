# ===============================
# MNIST LeNet-5 本地训练示例
# ===============================

import time
import os
import mindspore as ms
from mindspore import nn, context, Tensor
from mindspore.dataset import vision, transforms, MnistDataset
from mindspore.dataset.vision import Inter
from mindspore.common.initializer import Normal
from mindspore.train import Model
import matplotlib.pyplot as plt

# =====================
# 1. 运行环境配置
# =====================
# 本地 CPU 或 GPU
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")  # 如果有 GPU 改 "GPU"

# =====================
# 2. 数据处理
# =====================
def datapipe(path, batch_size=32):
    """构建 MNIST 数据集 pipeline"""
    # 图像处理操作
    image_ops = [
        vision.Resize((32, 32), interpolation=Inter.LINEAR),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    # 标签转换
    label_ops = transforms.TypeCast(ms.int32)

    ds = MnistDataset(path, shuffle=True)
    ds = ds.map(operations=image_ops, input_columns="image")
    ds = ds.map(operations=label_ops, input_columns="label")
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

# =====================
# 3. LeNet-5 模型
# =====================
class LeNet5(nn.Cell):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Dense(16*8*8, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_classes, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# =====================
# 4. 训练与评估函数
# =====================
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

def train_and_evaluate(name, opt_fn, train_path, test_path, epochs=3, batch_size=32):
    # 数据
    train_ds = datapipe(train_path, batch_size)
    test_ds = datapipe(test_path, batch_size)

    # 模型与优化器
    net = LeNet5()
    opt = opt_fn(net.trainable_params())
    model = Model(net, loss_fn, opt, metrics={"accuracy": nn.Accuracy()})

    print(f"\n===== 使用优化器：{name} =====")
    start_time = time.time()
    model.train(epochs, train_ds, dataset_sink_mode=False)
    cost = time.time() - start_time

    acc = model.eval(test_ds, dataset_sink_mode=False)["accuracy"]
    print(f"测试准确率：{acc*100:.2f}%")
    print(f"训练耗时：{cost:.2f}s")
    return {"optimizer": name, "accuracy": acc, "time": cost}

# =====================
# 5. 运行实验
# =====================
# 数据路径（请替换为本地 MNIST 解压路径）
train_path = "MNIST_Data/train"
test_path = "MNIST_Data/test"

# 检查路径是否存在
if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("请确保 MNIST 数据集已解压到 'MNIST_Data/train' 和 'MNIST_Data/test'")

results = []

# Momentum 优化器
results.append(train_and_evaluate(
    "Momentum",
    lambda p: nn.Momentum(p, learning_rate=0.01, momentum=0.9),
    train_path, test_path
))

# Adam 优化器 (学习率调低 + weight_decay)
results.append(train_and_evaluate(
    "Adam",
    lambda p: nn.Adam(p, learning_rate=0.0001, weight_decay=1e-4),
    train_path, test_path
))

# =====================
# 6. 可视化
# =====================
def plot_results(results):
    names = [r["optimizer"] for r in results]
    accs = [r["accuracy"]*100 for r in results]
    times = [r["time"] for r in results]
    x = range(len(names))

    # Accuracy 折线图
    plt.figure(figsize=(6,4))
    plt.plot(x, accs, marker='o', linewidth=2, markersize=8)
    plt.xticks(x, names)
    plt.ylim(90, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison")
    for i, v in enumerate(accs):
        plt.text(i, v+0.15, f"{v:.2f}%", ha='center', va='bottom', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Training Time 柱状图
    plt.figure(figsize=(6,4))
    colors = ['#1f77b4','#ff7f0e']
    bars = plt.bar(x, times, width=0.4, color=colors)
    plt.xticks(x, names)
    plt.ylim(0, max(times)*1.15)
    plt.ylabel("Training Time (s)")
    plt.title("Training Time Comparison")
    for i, v in enumerate(times):
        plt.text(i, v+0.5, f"{v:.2f}s", ha='center', va='bottom', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_results(results)
