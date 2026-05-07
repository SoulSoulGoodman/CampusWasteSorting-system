import torch
import torchvision.models as models
from torch import nn


def get_model(num_classes, pretrained=True):
    """
    构建并返回一个适用于特定分类任务（如垃圾分类）的 ResNet-18 模型。
    采用了“迁移学习”策略：保留预训练的特征提取层，仅替换最后的分类头。

    参数:
        num_classes (int): 你的业务场景需要输出的类别总数（例如 4 大类，或 40 种细分垃圾）。
        pretrained (bool): 是否加载官方在 ImageNet (千万级图片) 上预训练好的权重。
                           默认为 True，这相当于让模型站在巨人的肩膀上，收敛更快、准确率更高。

    返回:
        model: 改造完毕的 PyTorch 模型实例。
    """

    # 1. 加载骨干网络 (Backbone)
    # 实例化一个标准的 ResNet-18 深度残差网络。
    # 如果 pretrained=True，它会自带识别 1000 种通用物体的强大“先验知识”。
    model = models.resnet18(pretrained=pretrained)

    # 2. 获取原网络的特征维度
    # ResNet-18 默认的最后全连接层 (fc) 是接收 512 维的特征向量，输出 1000 个分类。
    # 我们需要知道这个 512 维的输入大小，以便后续做无缝衔接。
    in_features = model.fc.in_features

    # 3. 改造分类头 (Classification Head)
    # 核心操作！将原本输出 1000 类的全连接层，强行替换成一个新的全连接层 nn.Linear。
    # 新层的输入维度依然是 512 (in_features)，但输出维度变成了我们的垃圾类别数 (num_classes)。
    # 新层的权重是随机初始化的，后续训练（train.py）主要就是为了训练这一层。
    model.fc = nn.Linear(in_features, num_classes)

    # 4. 返回改造后的“定制版” AI 大脑
    return model