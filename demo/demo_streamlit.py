import streamlit as st
import socket, json, pandas as pd, os, time, io
from PIL import Image
from pathlib import Path
import db_manager

# 初始化引擎
db_manager.init_db()
st.set_page_config(page_title='校园垃圾智能分类平台', layout='wide', page_icon='♻️')

# 实时加载数据库内容
KNOWLEDGE_BASE = db_manager.get_knowledge_base()

if 'auth_status' not in st.session_state:
    st.session_state.update({'auth_status': False, 'user_id': None, 'username': None, 'role': None})

# =========================================
# 身份验证模块
# =========================================
if not st.session_state.auth_status:
    st.title('♻️ 校园垃圾智能分类平台')
    t1, t2 = st.tabs(["🔑 登录", "📝 注册"])
    with t1:
        u = st.text_input("用户名")
        p = st.text_input("密码", type='password')
        if st.button("进入系统", use_container_width=True):
            user = db_manager.login_user(u, p)
            if user:
                if user[3] == 'banned':
                    st.error("🚫 账号已封禁，请联系管理员")
                else:
                    st.session_state.update(
                        {'auth_status': True, 'user_id': user[0], 'username': user[1], 'role': user[2]})
                    st.rerun()
            else:
                st.error("账号或密码错误")
    with t2:
        nu, np = st.text_input("设置学号"), st.text_input("设置密码", type='password')
        nr = st.selectbox("角色申请", ["学生", "管理员"])
        if st.button("提交注册"):
            ok, msg = db_manager.register_user(nu, np, 'admin' if nr == "管理员" else 'user')
            st.success(msg) if ok else st.error(msg)
    st.stop()

# =========================================
# 侧边栏导航
# =========================================
with st.sidebar:
    st.header(f"👤 {st.session_state.username}")
    st.info(f"身份: {'管理员' if st.session_state.role == 'admin' else '普通学生'}")

    ann = db_manager.get_active_announcement()
    if ann: st.warning(f"📢 **系统公告**\n\n{ann}"); st.markdown('---')

    menu = ["📸 智能识别", "📂 识别足迹", "🌍 环保百科"]
    if st.session_state.role == 'admin': menu.append("🛡️ 管理后台")
    page = st.radio("导航菜单", menu, label_visibility="collapsed")

    st.markdown('---')
    if st.button("🚪 安全退出", use_container_width=True):
        st.session_state.auth_status = False;
        st.rerun()
    st.caption("开发者：胡昱璠 (24计科1班)")

# -----------------------------------------
# 页面 1：AI 推理
# -----------------------------------------
if page == "📸 智能识别":
    st.title('📸 AI 实时识别中心')
    up = st.file_uploader('请上传垃圾图片...', type=['jpg', 'png'])
    if up:
        img = Image.open(up).convert('RGB')
        c1, c2 = st.columns(2)
        with c1:
            st.image(img, use_container_width=True)
        with c2:
            if st.button('🚀 发送至 AI 云端', use_container_width=True, type='primary'):
                try:
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG')
                    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sk.connect(('127.0.0.1', 8888))
                    sk.sendall(buf.getvalue())
                    sk.shutdown(socket.SHUT_WR)
                    res = sk.recv(10240)
                    sk.close()
                    data = json.loads(res.decode('utf-8'))[0]

                    save_p = os.path.join(str(Path(__file__).resolve().parent.parent), 'demo', 'history_images',
                                          f"{int(time.time())}.jpg")
                    os.makedirs(os.path.dirname(save_p), exist_ok=True)
                    img.save(save_p)
                    db_manager.add_record(st.session_state.user_id, save_p, data[0], data[1], data[2])

                    st.success(f"🎯 AI 诊断结果：**{data[2]}**")
                    with st.expander("💡 官方投放指南", expanded=True):
                        st.markdown(KNOWLEDGE_BASE.get(data[2], "暂无内容"))
                except Exception as e:
                    st.error(f"服务器未响应: {e}")

# -----------------------------------------
# 页面 2：识别足迹 (含无限打回的拉锯申诉机制)
# -----------------------------------------
elif page == "📂 识别足迹":
    st.title('📂 我的历史记录')
    df = db_manager.get_history(st.session_state.user_id)
    if df.empty:
        st.info("暂无记录")
    else:
        df['显示结果'] = df['corrected_category'].fillna(df['category'])

        st.bar_chart(df['显示结果'].value_counts())
        st.divider()
        st.subheader("🖼️ 历史图像档案墙")

        cols = st.columns(4)
        for idx, row in df.iterrows():
            with cols[idx % 4]:
                with st.container(border=True):
                    if os.path.exists(row['image_path']):
                        st.image(row['image_path'], use_container_width=True)
                    else:
                        st.warning("图片文件已移除")

                    # ==========================================
                    # 场景 1：管理员已经修正过（可能对，也可能错）
                    # ==========================================
                    if pd.notna(row['corrected_category']):
                        st.success(f"✅ {row['corrected_category']} (管理员已修正)")

                        # 判断当前是否处于“二次申诉”中
                        if row.get('user_feedback') == 'pending':
                            st.button("⏳ 二次申诉核实中...", key=f"feed_{row['id']}", disabled=True)
                        else:
                            # 允许用户不服，再次发起申诉！
                            if st.button("🔄 还是错的？再次申诉", key=f"feed_{row['id']}",
                                         help="如果管理员修正依然有误，可再次打回"):
                                db_manager.submit_user_feedback(row['id'])  # 把状态重新变成 pending
                                st.toast("✅ 已发起二次申诉，工单已重新打回给管理员！")
                                time.sleep(0.5)
                                st.rerun()

                    # ==========================================
                    # 场景 2：还是 AI 的原始结果，尚未有人工干预
                    # ==========================================
                    else:
                        st.write(f"🤖 {row['category']}")

                        if row.get('user_feedback') == 'pending':
                            st.button("⏳ 异议审核中...", key=f"feed_{row['id']}", disabled=True)
                        else:
                            if st.button("🚩 结果不准？", key=f"feed_{row['id']}", help="提交给管理员人工核实"):
                                db_manager.submit_user_feedback(row['id'])
                                st.toast("✅ 已提交异议，管理员将尽快核实！")
                                time.sleep(0.5)
                                st.rerun()

                    st.caption(f"⏰ {row['created_at']}")

# -----------------------------------------
# 页面 3：动态百科
# -----------------------------------------
elif page == "🌍 环保百科":
    st.title("🌍 垃圾分类动态百科")
    st.info("内容由后勤处实时维护更新")
    for cat, content in KNOWLEDGE_BASE.items():
        with st.container(border=True): st.markdown(content)

# -----------------------------------------
# 页面 4：管理后台 (含 BI 大盘与多次纠错)
# -----------------------------------------
elif page == "🛡️ 管理后台" and st.session_state.role == 'admin':
    st.title("🛡️ 平台运维管理")

    # 【新增功能 2】：增加了一个全新的 BI 大盘 Tab
    t_bi, t1, t2, t3, t4, t5 = st.tabs(
        ["📊 校园环保大盘", "🗂️ 审核纠错", "👥 用户检索", "⚙️ 算法升级", "📢 公告管理", "📖 百科编辑"])

    # --- BI 数据大盘 ---
    with t_bi:
        st.subheader("🌍 全校减碳贡献实时看板")
        st.info("此数据根据全校师生累计识别次数与修正后的精准分类结果进行实时换算。")
        stats = db_manager.get_feedback_stats()

        col1, col2, col3 = st.columns(3)
        total_scans = int(stats['count'].sum()) if not stats.empty else 0

        # 换算公式 (模拟)：每次正确分类投递相当于减少 0.05kg 碳排放，22kg 等效于种一棵树
        col1.metric(label="全校总分类识别次数", value=f"{total_scans} 次", delta="活跃度良好")
        col2.metric(label="累计减碳量 (CO2e)", value=f"{total_scans * 0.05:.2f} kg", delta="环保贡献提升")
        col3.metric(label="地球等效植树量", value=f"{int(total_scans * 0.05 / 22)} 棵", delta="持续绿化中")

        st.markdown("### 📊 垃圾分类大盘分布")
        if not stats.empty:
            st.bar_chart(stats.set_index('category'))
        else:
            st.write("暂无分类数据积累。")

    # --- 审核纠错面板 ---
    with t1:
        with st.expander("📝 历史纠错记录"):
            st.dataframe(db_manager.get_logs('correction'), use_container_width=True, hide_index=True)

        hist = db_manager.get_history(is_admin=True)
        cols = st.columns(4)
        for idx, r in hist.head(28).iterrows():
            with cols[idx % 4]:
                # 如果用户对这条记录有异议，边框显示红色警告
                is_alert = (r.get('user_feedback') == 'pending')
                with st.container(border=True):
                    if is_alert:
                        st.error("🚩 用户标记此结果不准！")

                    # 快捷封禁
                    c1, c2 = st.columns([7, 3])
                    c1.caption(f"👤 **{r['username']}**")
                    if r['username'] != st.session_state.username:
                        if c2.button("🚫", key=f"q_{r['id']}"):
                            db_manager.update_user_status(r['user_id'], 'banned', st.session_state.username,
                                                          r['username'])
                            st.rerun()

                    if os.path.exists(r['image_path']): st.image(r['image_path'], use_container_width=True)

                    # 修正回显与再次修正入口
                    cur = r.get('corrected_category')
                    if pd.notna(cur):
                        st.success(f"已修正: {cur}")
                        d_idx = ["可回收物", "有害垃圾", "厨余垃圾", "其他垃圾"].index(cur)
                    else:
                        st.write(f"AI 预测: {r['category']}")
                        d_idx = 0

                    with st.expander("🛠️ 修正/处理异议"):
                        n = st.selectbox("改为", ["可回收物", "有害垃圾", "厨余垃圾", "其他垃圾"], index=d_idx,
                                         key=f"s_{r['id']}")
                        if st.button("提交修改", key=f"c_{r['id']}", type="primary"):
                            old_v = cur if pd.notna(cur) else r['category']
                            db_manager.correct_record(r['id'], n, old_v, st.session_state.username, r['username'])
                            st.toast("✅ 数据已更新");
                            time.sleep(0.5);
                            st.rerun()

    # --- 风控面板 ---
    with t2:
        with st.expander("📝 封禁日志"):
            st.dataframe(db_manager.get_logs('ban'), use_container_width=True)
        kw = st.text_input("🔍 搜索用户名...")
        usrs = db_manager.get_all_users(kw)
        for _, ur in usrs.iterrows():
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.write(f"👤 **{ur['username']}** ({ur['role']})")
            c2.write("✅ 正常" if ur['status'] == 'active' else "🚫 封禁")
            if ur['role'] != 'admin':
                label, target = ("封禁", "banned") if ur['status'] == 'active' else ("解封", "active")
                if c3.button(label, key=f"u_{ur['user_id']}"):
                    db_manager.update_user_status(ur['user_id'], target, st.session_state.username, ur['username'])
                    st.rerun()
            st.divider()

    # --- OTA 面板 ---
    with t3:
        with st.expander("📝 OTA 日志"): st.dataframe(db_manager.get_logs('ota'), use_container_width=True)
        f = st.file_uploader("上传 .pth")
        if f and st.button("确认覆盖"):
            p = os.path.join(str(Path(__file__).resolve().parent.parent), 'demo', 'model_40class.pth')
            with open(p, "wb") as w: w.write(f.getbuffer())
            db_manager.add_log('ota', st.session_state.username, f"更新模型: {f.name}")
            st.success("✅ 更新成功，请重启 server.py。")

    # --- 公告面板 ---
    with t4:
        st.subheader("发布公告")
        txt = st.text_area("内容")
        if st.button("广播"): db_manager.publish_announcement(txt); st.rerun()

    # --- CMS 面板 ---
    with t5:
        st.subheader("📖 动态百科 CMS")
        cat = st.selectbox("选择类别", list(KNOWLEDGE_BASE.keys()))
        new_c = st.text_area("编辑正文", value=KNOWLEDGE_BASE[cat], height=200)
        if st.button("💾 保存并发布", type="primary"):
            db_manager.update_knowledge(cat, new_c);
            st.rerun()