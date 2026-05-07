import sys
import os
import glob
import json

# 【工程技巧】动态修改环境变量，将 src 目录临时加入系统路径
# 这样即使在项目根目录运行此脚本，也能顺利导入 src 文件夹下的模块
sys.path.insert(0, 'src')

import torch
from model import get_model
from utils import load_image


def predict(model_path, classes_path, image_path, device='cpu', topk=1, label_map_path=None):
    """
    【专为批量压测定制的推理函数】
    与服务端单张推理的 predict.py 类似，但优化了数据返回格式（返回字典列表），
    使其更利于后续的大规模 JSON 序列化和 CSV 写入。
    """
    # 1. 读取基础类别字典
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    num_classes = len(classes)

    # 2. 模型初始化与权重挂载
    model = get_model(num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 开启评估模式，锁定 Batch Normalization 等动态层

    # 3. 图像预处理与张量转换
    tensor = load_image(image_path, device=device)

    # 4. 闭环推理（切断梯度追踪，最大化释放算力）
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        # 取出排名前 K 的置信度数值和对应索引
        topk_vals, topk_idx = torch.topk(probs, k=min(topk, num_classes))

    # 将张量转化为原生 Python 列表
    vals = topk_vals.tolist()
    idxs = topk_idx.tolist()
    results = []

    # 5. 加载二级分类映射字典 (用于将如 "易拉罐" 映射为 "可回收物")
    category_map = {}
    if label_map_path is None:
        default_map = os.path.join(os.path.dirname(__file__), 'label_map.json')
        if os.path.exists(default_map):
            label_map_path = default_map

    if label_map_path and os.path.exists(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as f:
            category_map = json.load(f)

    # 6. 数据拼装：将预测结果打包成结构化的字典
    for j, idx in enumerate(idxs):
        label = classes[int(idx)]
        prob = float(vals[j])
        category = category_map.get(label, 'other') if category_map else 'other'
        results.append({'label': label, 'prob': prob, 'category': category})

    return results


if __name__ == '__main__':
    """
    批量压测主控程序入口
    工作流：扫描图片文件夹 -> 遍历送入 AI 引擎 -> 聚合结果 -> 生成测试报表
    """
    # ==========================================
    # 1. 核心资源路径配置
    # ==========================================
    model_path = os.path.join('demo', 'demo_model.pth')
    classes_path = os.path.join('demo', 'classes.json')
    label_map = os.path.join('demo', 'label_map.json')
    # 测试集图像所在目录
    img_dir = os.path.join('demo', 'images')

    # 硬件自适应诊断
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ==========================================
    # 2. 自动化文件扫描
    # ==========================================
    # 使用 glob 模块通配符匹配，自动抓取目录下所有的 .jpg 图片
    paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if not paths:
        print('❌ 扫描失败：在', img_dir, '目录中未找到任何图片数据')
        sys.exit(1)  # 异常退出程序

    all_results = {}

    # ==========================================
    # 3. 开启批量推理流水线
    # ==========================================
    print(f"🚀 开始执行自动化批量压测，共计 {len(paths)} 个测试样本...")
    for p in paths:
        # 逐张执行预测，这里 topk=1 表示我们只关心模型的“第一直觉”判断
        res = predict(model_path, classes_path, p, device=device, topk=1, label_map_path=label_map)

        # 提取图片文件的纯文件名（如 001.jpg）作为标识 key
        name = os.path.basename(p)
        all_results[name] = res

        # 在终端实时打印测试流水，方便开发者监控进度
        print(f"📄 正在推断: {name}")
        for r in res:
            print(f"  └─ 模型标签: {r['label']} | 置信度: {r['prob']:.4f} | 国标归属: {r['category']}")
        print("-" * 30)

    # ==========================================
    # 4. 数据沉淀与报表生成 (极其重要的工程落地环节)
    # ==========================================

    # 【方案一】：生成 JSON 数据湖文件
    # 作用：方便未来二次开发，比如供其他脚本读取统计准确率，或者被前端读取做可视化展现
    out_path = os.path.join('demo', 'batch_predictions.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        # indent=2 使生成的 JSON 文件具有良好的层级缩进，人类可读
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print('✅ [阶段一] JSON 数据湖已持久化至:', out_path)

    # 【方案二】：生成 CSV 统计报表 (答辩/论文材料来源)
    # 作用：可以用 Excel 直接打开，用于制作论文中的“模型准确率统计表”或“Top-1 精度分析图”
    csv_path = os.path.join('demo', 'batch_predictions.csv')
    import csv

    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        # 写入标准的二维表头
        writer.writerow(['图像名称', '置信度排名', 'AI预测具体标签', '置信度(%)', '所属分类'])

        # 扁平化展开嵌套的字典数据并写入表格
        for fname, entries in all_results.items():
            for rank, e in enumerate(entries, start=1):
                writer.writerow([fname, rank, e['label'], f"{e['prob']:.6f}", e['category']])

    print('✅ [阶段二] CSV 自动化评审报表已生成至:', csv_path)
    print('🎉 批量压测任务圆满结束！')