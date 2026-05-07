import sqlite3
import hashlib
import pandas as pd
from pathlib import Path
import os

# 动态定位数据库路径
DB_PATH = os.path.join(str(Path(__file__).resolve().parent.parent), 'history.db')


def init_db():
    """初始化数据库架构，支持多版本平滑升级字段并强力填充初始数据"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 用户表
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS users
                   (
                       user_id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       username
                       TEXT
                       UNIQUE
                       NOT
                       NULL,
                       password_hash
                       TEXT
                       NOT
                       NULL,
                       role
                       TEXT
                       DEFAULT
                       'user',
                       status
                       TEXT
                       DEFAULT
                       'active'
                   )
                   ''')
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'active'")
    except:
        pass

    # 2. 识别流水表 (包含 corrected_category 和 user_feedback)
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS scan_history
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       user_id
                       INTEGER
                       NOT
                       NULL,
                       image_path
                       TEXT,
                       prediction
                       TEXT,
                       probability
                       REAL,
                       category
                       TEXT,
                       corrected_category
                       TEXT,
                       user_feedback
                       TEXT,
                       created_at
                       DATETIME
                       DEFAULT
                       CURRENT_TIMESTAMP
                   )
                   ''')
    try:
        cursor.execute("ALTER TABLE scan_history ADD COLUMN corrected_category TEXT")
    except:
        pass

    # 【新增】：追加用户异议反馈字段
    try:
        cursor.execute("ALTER TABLE scan_history ADD COLUMN user_feedback TEXT")
    except:
        pass

    # 3. 全局公告表
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS announcements (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL, is_active INTEGER DEFAULT 1, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)')

    # 4. 环保百科 (CMS 动态内容库)
    cursor.execute('CREATE TABLE IF NOT EXISTS encyclopedia (category TEXT PRIMARY KEY, content TEXT NOT NULL)')

    # 强力初始化百科
    cursor.execute("SELECT COUNT(*) FROM encyclopedia")
    if cursor.fetchone()[0] == 0:
        default_kb = [
            ("可回收物", "♻️ **可回收物**：废纸张、废塑料、废金属等。\n\n**投放要求**：清洁干燥，避免污染。"),
            ("有害垃圾", "☣️ **有害垃圾**：废电池、灯管、药品等。\n\n**投放要求**：轻放，密封，防止破损。"),
            ("厨余垃圾", "🍏 **厨余垃圾**：食材废料、剩菜等。\n\n**投放要求**：沥干水分，去除包装投放。"),
            ("其他垃圾", "🗑️ **其他垃圾**：卫生纸、烟蒂、碎瓷等。\n\n**投放要求**：尽量沥干水分，严禁混入。")
        ]
        cursor.executemany("INSERT INTO encyclopedia (category, content) VALUES (?, ?)", default_kb)

    # 5. 系统审计日志表
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS system_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, action_type TEXT NOT NULL, operator TEXT NOT NULL, detail TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)')

    conn.commit()
    conn.close()


# ==========================================
# 审计日志模块
# ==========================================
def add_log(action_type, operator, detail):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO system_logs (action_type, operator, detail) VALUES (?, ?, ?)",
                   (action_type, operator, detail))
    conn.commit()
    conn.close()


def get_logs(action_type):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT operator AS 操作人, detail AS 操作详情, created_at AS 操作时间 FROM system_logs WHERE action_type = ? ORDER BY id DESC"
    df = pd.read_sql(query, conn, params=(action_type,))
    conn.close()
    return df


# ==========================================
# 用户与风控模块
# ==========================================
def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute('SELECT user_id, username, role, status FROM users WHERE username=? AND password_hash=?',
                   (username, pwd_hash))
    user = cursor.fetchone()
    conn.close()
    return user


def register_user(username, password, role='user'):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        cursor.execute('INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)', (username, pwd_hash, role))
        conn.commit()
        return True, "注册成功"
    except:
        return False, "用户名已存在"
    finally:
        conn.close()


def get_all_users(search_term=""):
    conn = sqlite3.connect(DB_PATH)
    if search_term:
        query = "SELECT user_id, username, role, status FROM users WHERE username LIKE ? ORDER BY user_id DESC"
        df = pd.read_sql(query, conn, params=(f'%{search_term}%',))
    else:
        df = pd.read_sql("SELECT user_id, username, role, status FROM users ORDER BY user_id DESC", conn)
    conn.close()
    return df


def update_user_status(user_id, new_status, operator, target):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET status = ? WHERE user_id = ?", (new_status, user_id))
    conn.commit()
    conn.close()
    add_log('ban', operator, f"对用户【{target}】执行了 {'封禁' if new_status == 'banned' else '解封'}")


# ==========================================
# 业务流水、纠错与 BI 统计模块
# ==========================================
def add_record(user_id, image_path, prediction, probability, category):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO scan_history (user_id, image_path, prediction, probability, category) VALUES (?, ?, ?, ?, ?)",
        (user_id, image_path, prediction, probability, category))
    conn.commit()
    conn.close()


def get_history(user_id=None, is_admin=False):
    conn = sqlite3.connect(DB_PATH)
    if is_admin:
        query = "SELECT h.*, u.username FROM scan_history h JOIN users u ON h.user_id = u.user_id ORDER BY h.id DESC"
        df = pd.read_sql(query, conn)
    else:
        query = "SELECT * FROM scan_history WHERE user_id = ? ORDER BY id DESC"
        df = pd.read_sql(query, conn, params=(user_id,))
    conn.close()
    return df


def correct_record(record_id, new_val, old_val, operator, target):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 纠错完成后，自动将用户的异议状态重置为 resolved
    cursor.execute("UPDATE scan_history SET corrected_category = ?, user_feedback = 'resolved' WHERE id = ?",
                   (new_val, record_id))
    conn.commit()
    conn.close()
    add_log('correction', operator, f"修正流水({record_id}): [{old_val}] -> [{new_val}]")


# 【新增】：用户提交异议申请
def submit_user_feedback(record_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE scan_history SET user_feedback = 'pending' WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()


# 【新增】：BI 看板数据聚合
def get_feedback_stats():
    conn = sqlite3.connect(DB_PATH)
    # 使用 IFNULL：如果有管理员的人工修正结果，优先按人工结果统计；否则按 AI 结果统计，保证大盘数据极其准确
    query = """
            SELECT IFNULL(corrected_category, category) AS category, COUNT(*) as count
            FROM scan_history
            GROUP BY IFNULL(corrected_category, category) \
            """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# ==========================================
# CMS 与公告模块
# ==========================================
def get_knowledge_base():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT category, content FROM encyclopedia", conn)
    conn.close()
    return dict(zip(df['category'], df['content']))


def update_knowledge(category, content):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE encyclopedia SET content = ? WHERE category = ?", (content, category))
    conn.commit()
    conn.close()


def publish_announcement(content):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE announcements SET is_active = 0")
    if content.strip(): cursor.execute("INSERT INTO announcements (content) VALUES (?)", (content,))
    conn.commit()
    conn.close()


def get_active_announcement():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM announcements WHERE is_active = 1 ORDER BY id DESC LIMIT 1")
    res = cursor.fetchone()
    conn.close()
    return res[0] if res else None