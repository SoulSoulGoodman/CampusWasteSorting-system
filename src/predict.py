import argparse
import json
import os

import torch
from PIL import Image

# 从你自己的模块中导入网络结构定义和预处理工具
from model import get_model
from utils import load_image


def predict(model_path, image_path, topk=3, device='cpu', classes_path=None, label_map_path=None):
    """
    核心推理函数：负责加载模型、处理单张图片、进行前向传播并返回分类结果。
    """
    # ==========================================
    # 1. 硬件加速智能检测 (覆盖传入的 device 参数)
    # ==========================================
    if torch.cuda.is_available():
        device = 'cuda'  # 优先唤醒 NVIDIA 显卡 (CUDA 计算节点)
    elif torch.backends.mps.is_available():
        device = 'mps'  # 针对 Mac Apple Silicon 芯片的底层硬件加速引擎
    else:
        device = 'cpu'  # 降级使用通用 CPU 算力

    print(f"[INFO] 当前使用的推理计算设备为: {device}")

    # ==========================================
    # 2. 词典加载 (解析一级类别标签)
    # ==========================================
    if classes_path is None:
        # 如果没传，默认去项目根目录下的 demo 文件夹找 classes.json
        classes_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo', 'classes.json')

    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    num_classes = len(classes)  # 动态获取模型的输出分类总数 (例如 4 或 40)

    # ==========================================
    # 3. AI 模型挂载与状态恢复
    # ==========================================
    # 实例化网络结构图纸 (禁用预训练，因为我们要加载自己训练好的权重)
    model = get_model(num_classes, pretrained=False).to(device)
    # 反序列化 .pth 文件，将训练好的权重参数 (state_dict) 注入到网络结构中
    model.load_state_dict(torch.load(model_path, map_location=device))
    # 【关键】将模型切换至评估模式 (Evaluation Mode)
    # 这会关闭 Dropout 和 Batch Normalization 的动态更新，保证推理结果的稳定性
    model.eval()

    # ==========================================
    # 4. 图像预处理与张量转换
    # ==========================================
    # 将物理图片转化为深度学习可读的 Tensor (张量)，并推送到对应的计算设备上
    tensor = load_image(image_path, device=device)

    # ==========================================
    # 5. 前向传播 (Forward Pass) 与预测
    # ==========================================
    # 【关键】使用 torch.no_grad() 禁用梯度计算引擎
    # 既然是推理阶段，不需要反向传播更新参数，禁用它能大幅节省内存和算力开销
    with torch.no_grad():
        outputs = model(tensor)  # 输出的是 Raw Logits (未归一化的原始得分)
        # 沿分类维度 (dim=1) 执行 Softmax 激活函数，将得分转化为总和为 1 的概率分布 (置信度)
        probs = torch.softmax(outputs, dim=1)[0]
        # 提取概率最高的前 K 个预测结果及其对应的索引
        topk_vals, topk_idx = torch.topk(probs, k=min(topk, num_classes))

    # 将处于显存/内存中的 Tensor 转化为普通的 Python 列表
    topk_vals_list = topk_vals.tolist()
    topk_idx_list = topk_idx.tolist()

    # 利用之前的字典，将数字索引翻译成人类可读的标签 (如: "易拉罐")
    results = [(classes[int(idx)], float(topk_vals_list[j])) for j, idx in enumerate(topk_idx_list)]

    # ==========================================
    # 6. 二级类别映射 (如: "易拉罐" -> "可回收物")
    # ==========================================
    category_map = {}
    if label_map_path is None:
        default_map = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo', 'label_map.json')
        if os.path.exists(default_map):
            label_map_path = default_map

    if label_map_path and os.path.exists(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as f:
            category_map = json.load(f)

    final_results = []
    # 遍历 Top-K 结果，进行最终的拼装
    for label, prob in results:
        # 如果存在映射字典，就去查四大类归属；如果没有，就默认划归为 'other'
        category = category_map.get(label, 'other') if category_map else 'other'
        final_results.append((label, prob, category))

    return final_results  # 返回格式: [("具体类别", 0.98, "四大类"), ...]


if __name__ == '__main__':
    """
    独立测试入口：允许开发者通过命令行直接调用此脚本进行测试，而无需启动庞大的后端服务器。
    """
    # 初始化命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='训练好的模型权重文件 (.pth) 路径')
    parser.add_argument('--image', required=True, help='待预测的物理图片路径')
    parser.add_argument('--classes', required=False, help='一级字典 (classes.json) 路径')
    parser.add_argument('--label-map', required=False, help='二级映射字典 (label_map.json) 路径')
    parser.add_argument('--topk', type=int, default=1, help='输出前几个最可能的预测结果')
    parser.add_argument('--single', action='store_true', help='开启后，只打印出最终的四大类归属 (精简模式)')
    args = parser.parse_args()

    # 备用硬件检测 (实际会在 predict 内部被再次覆盖)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 发起预测请求
    res = predict(args.model, args.image, args.topk, device, classes_path=args.classes, label_map_path=args.label_map)

    # 根据命令行参数决定终端的打印格式
    if args.single:
        if res:
            # 只提取 Top-1 结果的四大类归属
            top_label, top_prob, top_category = res[0]
            print(top_category)
        else:
            print('unknown')
    else:
        # 详细打印所有 Top-K 结果的溯源信息
        for label, prob, category in res:
            print(f"{label}: {prob:.4f}  -> {category}")