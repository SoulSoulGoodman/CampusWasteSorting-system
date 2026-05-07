from torchvision import transforms
from PIL import Image
import torch


def get_transforms(train=False):
    """
    获取图像预处理与数据增强（Data Augmentation）流水线。
    根据当前是“训练模式”还是“推理/验证模式”，返回不同的处理逻辑。
    """
    if train:
        # ==========================================
        # 🏋️‍♂️ 训练集专属：数据增强流水线 (防止过拟合)
        # ==========================================
        return transforms.Compose([
            # 1. 随机裁剪并缩放至 224x224
            # 迫使模型在图片只露出局部（比如只看到半个易拉罐）时也能认出物体，增强泛化能力
            transforms.RandomResizedCrop(224),

            # 2. 随机水平翻转 (50% 概率)
            # 让模型明白：一个向左的瓶子和一个向右的瓶子，都是瓶子
            transforms.RandomHorizontalFlip(),

            # 3. 转化为张量 (Tensor)
            # 将 PIL 格式的图片 (像素值 0-255) 转化为 PyTorch 可计算的浮点数张量，并将数值归一化到 0~1 之间
            # 同时将图片维度从 [H, W, C] (高,宽,通道) 调整为 [C, H, W]
            transforms.ToTensor(),

            # 4. 标准化 (Normalize)
            # 极其重要！使用 ImageNet 数据集的全局均值(mean)和标准差(std)对图像进行标准化。
            # 让数据分布以 0 为中心，加快神经网络梯度下降的收敛速度。
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        # ==========================================
        # 🔍 验证/推理集专属：标准化格式流水线
        # ==========================================
        return transforms.Compose([
            # 1. 统一等比例缩放，使得短边为 256 像素
            # 为什么不直接缩放到 224？为了保留物体的长宽比，防止图片被无情拉伸变形
            transforms.Resize(256),

            # 2. 从图像正中心裁剪出 224x224 的区域
            # 假设主体大多位于画面中央，剔除边缘无用的背景信息
            transforms.CenterCrop(224),

            transforms.ToTensor(),

            # 保持与训练集完全一致的标准化参数
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def load_image(path, device='cpu'):
    """
    加载单张物理图片，并将其转化为可以直接送入神经网络推理的张量格式。

    参数:
        path (str): 本地图片的绝对或相对路径
        device (str): 计算设备 (cpu, cuda, mps)，将数据直接加载到对应的显存/内存中
    返回:
        tensor: 形状为 [1, 3, 224, 224] 的图像张量
    """
    # 1. 读取图像并强制转化为 RGB 三通道格式
    # 这一步极其关键，防止用户上传包含透明通道的 PNG (4通道) 或黑白灰度图 (1通道) 导致程序崩溃
    img = Image.open(path).convert('RGB')

    # 2. 获取推理模式下的预处理流水线
    t = get_transforms(train=False)

    # 3. 核心张量变形
    # t(img) 会将图片变为形状为 [3, 224, 224] 的张量。
    # 但是神经网络默认接收“批次 (Batch)”数据，即要求格式为 [Batch_size, Channels, Height, Width]。
    # 所以必须调用 unsqueeze(0) 在最前面增加一个维度，将形状变为 [1, 3, 224, 224]。
    # 最后 .to(device) 将这块内存数据转移到指定的加速芯片上。
    tensor = t(img).unsqueeze(0).to(device)

    return tensor