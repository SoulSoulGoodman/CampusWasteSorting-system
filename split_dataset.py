#!/usr/bin/env python3
import os
import shutil
import random

# 随机种子保持一致
random.seed(42)

source_dir = 'dataset_raw'
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# 创建目录
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    if os.path.isdir(category_path):
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)

# 分割数据
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    if not os.path.isdir(category_path):
        continue
    
    images = os.listdir(category_path)
    random.shuffle(images)
    
    # 8:2 分割
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # 复制到 train 目录
    for img in train_images:
        src = os.path.join(category_path, img)
        dst = os.path.join(train_dir, category, img)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    
    # 复制到 val 目录
    for img in val_images:
        src = os.path.join(category_path, img)
        dst = os.path.join(val_dir, category, img)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    
    print(f"{category}: {len(train_images)} train, {len(val_images)} val")

print("\n数据集分割完成！")
print(f"Train: dataset/train")
print(f"Val: dataset/val")
