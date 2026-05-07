
---

# ♻️ 校园垃圾智能分类与监测系统 (Campus Garbage Classification System)

本项目是一个基于 C/S（客户端/服务端）架构构建的现代化校园垃圾分类辅助决策系统。系统核心采用 PyTorch 框架下的 ResNet-18 深度残差网络进行图像特征提取与分类推理，支持 40 类国标垃圾的高精度识别，并自动映射至四大类（可回收物、有害垃圾、厨余垃圾、其他垃圾）。

为了应对校园高峰期的高并发访问，系统采用了**前后端分离架构**与**主从多线程异步调度模型**，并在表现层集成了数据可视化看板，旨在为智慧校园的环保建设提供一站式解决方案。

## 🛠️ 核心技术栈

* **表现层 (Frontend)：** Streamlit (提供响应式 Web UI 与 Pandas 数据可视化看板)
* **传输层 (Network)：** 纯 Python 内置 `socket` 模块 (TCP 全双工通信隧道，自研防粘包机制)
* **业务逻辑层 (Backend AI)：** PyTorch (ResNet-18 预训练模型)、`threading` 多线程并发控制
* **数据层 (Database)：** SQLite3 关系型数据库 (持久化识别历史与分类置信度)、本地文件归档
* **硬件加速支持：** 自动检测并支持 NVIDIA CUDA 及 **Apple Silicon (MPS)** 原生加速。

## 📂 项目目录结构

```text
├── dataset/                 # 数据集存放目录 (需自行下载解压)
│   ├── train/               # 训练集
│   └── val/                 # 验证集
├── demo/
│   ├── demo_streamlit.py    # 表现层：Streamlit Web 前端主程序
│   ├── classes_40.json      # 40分类模型类别字典
│   └── label_map_40.json    # 40分类至四大类的映射规则字典
├── src/
│   ├── server.py            # 业务层：AI 服务端主程序 (负责 Socket 监听与推理)
│   ├── train.py             # 模型训练脚本
│   ├── predict.py           # 命令行推理测试脚本
│   ├── model.py             # ResNet-18 网络拓扑定义
│   └── utils.py             # 图像预处理与张量转换工具类
├── requirements.txt         # 项目依赖清单
├── README.md                # 项目说明文档 (当前文件)
└── history.db               # (自动生成) SQLite 历史数据库
```

## 🚀 快速启动指南

### 1. 环境准备 (Environment Setup)
推荐使用虚拟环境进行环境隔离，以保证团队开发环境的一致性。

```bash
# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows 用户请使用 .venv\Scripts\activate

# 安装核心依赖
pip install -r requirements.txt
```

### 2. 本地全链路运行 (Run the System)
本系统采用前后端解耦设计，需要**同时启动两个终端窗口**。

**终端 1：启动 AI 算力服务端 (后厨)**
```bash
python src/server.py
# 成功标志: [INFO] 🚀 AI 服务器已启动 (硬件加速: mps/cuda/cpu)，监听端口 8888...
```

**终端 2：启动 Web 交互前端 (前台)**
```bash
streamlit run demo/demo_streamlit.py
# 成功标志: 浏览器自动弹出系统主页 (默认地址 http://localhost:8501)
```

## 🧠 模型训练与进阶配置 (Model Training)

如果你需要针对校园特定垃圾（如外卖盒、特定品牌的饮料瓶）微调模型，或者使用开源的 40 类大型数据集（如华为云/百度飞桨数据集），请按照以下步骤重新训练模型：

### 获取 40 类推荐数据集
1. 在 Kaggle 或百度 AI Studio 搜索 "Garbage Classification 40 Classes" 进行下载。
2. 解压至 `dataset/` 目录，确保结构为 `dataset/train/易拉罐/img.jpg` 等格式。

### 执行训练脚本
```bash
python src/train.py \
    --data-dir dataset \
    --epochs 10 \
    --batch-size 32 \
    --save-path model_40.pth
```
*注：训练结束后会自动生成权重文件 `model_40.pth` 和对应的 `classes.json`。*

### 独立命令行推理 (Debug 用)
如需绕过前端直接测试模型准确率，可使用内置的预测脚本：
```bash
python src/predict.py \
    --model model_40.pth \
    --image path/to/test_img.jpg \
    --classes demo/classes_40.json \
    --label-map demo/label_map_40.json \
    --single
```


