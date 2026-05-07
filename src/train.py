import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets

# 导入你自定义的模型结构和预处理工具
from model import get_model
from utils import get_transforms


def train(data_dir, epochs, batch_size, lr, save_path, device):
    """
    核心训练函数：定义了数据加载、前向传播、反向传播以及模型验证的完整生命周期。
    """
    # ==========================================
    # 1. 数据集挂载与流水线构建 (Data Loading)
    # ==========================================
    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'

    # 使用 ImageFolder 自动解析目录。它极其聪明，会把子文件夹的名字（如"可回收物"）自动转成数字标签
    train_ds = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    val_ds = datasets.ImageFolder(val_dir, transform=get_transforms(train=False))

    # DataLoader 是搬运工：把零散的图片打包成批次 (batch_size)，打乱顺序 (shuffle) 送给显卡
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ==========================================
    # 2. 组装 AI 大脑：模型、损失函数与优化器
    # ==========================================
    num_classes = len(train_ds.classes)
    # 实例化模型，并推送到对应的硬件加速器（CPU/GPU/MPS）上
    model = get_model(num_classes, pretrained=False).to(device)

    # 交叉熵损失 (CrossEntropyLoss)：多分类任务的标准裁判，衡量预测结果和标准答案的差距
    criterion = nn.CrossEntropyLoss()
    # Adam 优化器：负责根据裁判给出的差距（梯度），自动调整学习步伐，更新模型权重
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0  # 记录历史最高验证集准确率
    print("✅ 数据集加载完毕！马上开始跑第一轮（可能需要稍等片刻）...")

    # ==========================================
    # 3. 核心训练与验证大循环
    # ==========================================
    for epoch in range(1, epochs + 1):

        # ------------------ [训练阶段] ------------------
        model.train()  # 开启训练模式，激活 Dropout 和 BatchNorm 层
        running = 0
        correct = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 第一步：清空上一轮的残余梯度，防止污染（极其重要！）
            optimizer.zero_grad()

            # 第二步：前向传播（让模型做题）
            outputs = model(imgs)

            # 第三步：计算损失（看看错得多离谱）
            loss = criterion(outputs, labels)

            # 第四步：反向传播（计算责任分摊，求梯度）
            loss.backward()

            # 第五步：权重更新（吸取教训，修改参数）
            optimizer.step()

            # 统计训练准确率
            running += imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

            if running % (batch_size * 50) == 0:
                print(f"  正在训练... 已处理 {running} 张图片")

        train_acc = correct / running if running else 0

        # ------------------ [验证阶段] ------------------
        model.eval()  # 切换到评估模式，关闭随机性，保证测试公平
        val_running = 0
        val_correct = 0

        # 考试不需要复习（不更新参数），切断梯度追踪可节省大量显存并加速
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_running += imgs.size(0)
                val_correct += (preds == labels).sum().item()

        val_acc = val_correct / val_running if val_running else 0

        print(f"📈 Epoch {epoch}/{epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        # ==========================================
        # 4. 最优模型持久化落盘 (Checkpoint)
        # ==========================================
        # 只有当本轮考试成绩打破历史记录时，才保存。防止模型过度训练变“笨”（过拟合）
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存模型的灵魂（权重矩阵）
            torch.save(model.state_dict(), save_path)
            # 同步保存类别密码本，防止推理时数字和汉字对不上号
            with open('classes.json', 'w', encoding='utf-8') as f:
                json.dump(train_ds.classes, f, ensure_ascii=False)


def parse_args():
    """
    配置命令行参数解析器：让你在终端敲代码时可以灵活传参，不用改源文件。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help="数据集根目录")
    parser.add_argument('--epochs', type=int, default=10, help="总共训练多少轮")
    parser.add_argument('--batch-size', type=int, default=32, help="每次喂给显卡多少张图")
    parser.add_argument('--lr', type=float, default=1e-4, help="学习率（迈步大小）")
    parser.add_argument('--save-path', default='model.pth', help="训练好的模型存到哪里")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 智能硬件侦测系统
    # 检查是否有英伟达显卡
    if torch.cuda.is_available():
        device = 'cuda'
    # 重点：检查是不是苹果 M 系列芯片 (开启 MPS 硬件加速)
    elif torch.backends.mps.is_available():
        device = 'mps'
    # 如果都不是，才用 CPU 进行保底计算
    else:
        device = 'cpu'

    print("=" * 50)
    print(f"🚀 AI 训练任务已启动")
    print(f"⚙️  当前激活的计算设备是: [{device.upper()}]")
    print("=" * 50)

    # 启动炼丹
    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.save_path, device)