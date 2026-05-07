import socket
import threading
import json
import os
import tempfile
import torch
from pathlib import Path

# 引入自定义的预测模块，实现业务逻辑解耦
from predict import predict

# --- 1. 智能硬件加速检测 ---
# 根据运行环境自动选择最优算力芯片
if torch.cuda.is_available():
    DEVICE = 'cuda'  # NVIDIA 显卡加速 (Windows/Linux)
elif torch.backends.mps.is_available():
    DEVICE = 'mps'  # Apple Silicon (M1/M2/M3/M4) 原生硬件加速
else:
    DEVICE = 'cpu'  # 通用 CPU 计算 (兜底方案)

# ==========================================
# ⚠️ 核心配置区：路径自动化解析
# ==========================================
# 使用 pathlib 动态获取项目根目录，增强代码在不同电脑上的移植性
root_dir = str(Path(__file__).resolve().parent.parent)

# 模型路径：指向训练好的权重文件
MODEL_PATH = os.path.join(root_dir, 'demo', 'model_40class.pth')

# 一级字典：将模型输出的数字索引 (0,1,2...) 映射为标签字符串 ("0","1"...)
CLASSES_PATH = os.path.join(root_dir, 'classes.json')

# 二级映射字典：将内部标签翻译为最终的中文分类名称 (如 "有害垃圾")
LABEL_MAP_PATH = os.path.join(root_dir, 'demo', 'label_map.json')


def handle_client(conn, addr):
    """
    【并发处理函数】：每个连接都会开启一个独立的子线程执行此函数。
    实现了“一客一议”，保证多个用户同时上传图片时服务器不会卡死。
    """
    print(f"[INFO] 📡 收到来自 {addr} 的连接请求, 启动推理子线程...")
    try:
        # 1. 接收图片数据流：循环读取直到客户端关闭发送
        data = bytearray()
        while True:
            packet = conn.recv(4096)  # 每次读取 4KB
            if not packet:
                break
            data.extend(packet)

        if not data:
            return

        # 2. 内存数据落地：将二进制流存入临时文件，供 AI 引擎读取
        # 使用 tempfile 避免文件名冲突，并确保推理完成后自动清理
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(data)
            temp_path = temp_file.name

        print(f"[INFO] 📥 图片已缓存至临时目录，开始调用 ResNet 引擎...")

        # 3. 核心推理：调用 predict.py 中的推理逻辑
        # topk=4 表示返回概率最高的前 4 个选项
        results = predict(
            model_path=MODEL_PATH,
            image_path=temp_path,
            topk=4,
            device=DEVICE,
            classes_path=CLASSES_PATH,
            label_map_path=LABEL_MAP_PATH
        )

        # 4. 磁盘清理：删除临时图片，防止服务器存储空间泄露
        os.remove(temp_path)

        # 5. 结果回传：将 Python 对象序列化为 JSON 字符串并发送
        response_json = json.dumps(results)
        conn.sendall(response_json.encode('utf-8'))
        print(f"[INFO] 📤 推理完成，JSON 结果已回传至客户端。")

    except Exception as e:
        print(f"[ERROR] ❌ 业务处理异常: {e}")
    finally:
        # 无论成功失败，必须关闭 Socket 连接，释放网络资源
        conn.close()


def start_server():
    """
    【主线程】：负责初始化网络端口并循环监听客户端请求。
    """
    # 建立 IPv4 + TCP 协议的套接字
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 端口复用配置：解决服务器频繁重启时产生的 "Address already in use" 报错
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 绑定本地 IP 与 8888 端口
    server.bind(('127.0.0.1', 8888))

    # 开始监听：允许最大 10 个请求在队列中等待
    server.listen(10)

    print("=" * 60)
    print(f"🚀 AI 分布式算力中枢已成功启动")
    print(f"⚙️  当前激活硬件加速引擎: [{DEVICE.upper()}]")
    print(f"🗂️  模型挂载: {os.path.basename(MODEL_PATH)}")
    print(f"🎧 正在监听端口 8888，等待客户端调度...")
    print("=" * 60)

    while True:
        # 阻塞式等待：直到有客户端连接进来
        conn, addr = server.accept()

        # 核心多线程架构：为新用户分配一个“服务员”线程，主线程立即回去等待下个用户
        client_thread = threading.Thread(target=handle_client, args=(conn, addr))
        client_thread.daemon = True  # 设置为守护线程，主程序退出时自动销毁
        client_thread.start()


if __name__ == '__main__':
    # 程序的执行入口
    start_server()