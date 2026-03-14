import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

def ask_question(question, history):
    if not question.strip():
        return history, ""
    
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "抱歉，无法获取回答")
        else:
            error_detail = response.json().get("detail", "未知错误")
            answer = f"错误: {error_detail}"
    except requests.exceptions.Timeout:
        answer = "请求超时，请稍后重试"
    except requests.exceptions.ConnectionError:
        answer = "无法连接到后端服务，请确保后端已启动"
    except Exception as e:
        answer = f"发生错误: {str(e)}"
    
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return history, ""

def rebuild_knowledge_base():
    try:
        response = requests.post(
            f"{API_URL}/init?force_rebuild=true",
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("message", "知识库重建成功")
        else:
            error_detail = response.json().get("detail", "未知错误")
            return f"错误: {error_detail}"
    except requests.exceptions.Timeout:
        return "请求超时，重建可能仍在进行中"
    except requests.exceptions.ConnectionError:
        return "无法连接到后端服务"
    except Exception as e:
        return f"发生错误: {str(e)}"

def check_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return "✅ 后端服务正常运行"
        else:
            return "❌ 后端服务异常"
    except:
        return "❌ 无法连接到后端服务"

custom_css = """
:root {
    --primary: #0057ff;
    --primary-hover: #004ad9;
    --primary-pressed: #003db3;
    --primary-transparent-1: rgba(0,87,255,0.06);
    --primary-transparent-2: rgba(0,87,255,0.1);
    --primary-transparent-3: rgba(0,87,255,0.15);
    --text-primary: #000000;
    --text-secondary: rgba(0,0,0,0.85);
    --text-tertiary: rgba(0,0,0,0.5);
    --text-quaternary: rgba(0,0,0,0.3);
    --bg-primary: #ffffff;
    --bg-secondary: #f9fafb;
    --bg-tertiary: #f3f4f6;
    --border-primary: rgba(0,0,0,0.2);
    --border-secondary: rgba(0,0,0,0.12);
    --border-tertiary: rgba(0,0,0,0.08);
    --shadow-lv1: 0px 1px 2px 0px rgba(0,0,0,0.05), 0px 0px 1px 0px rgba(0,0,0,0.15);
    --shadow-lv2: 0px 6px 10px 0px rgba(0,0,0,0.08), 0px 0px 1px 0px rgba(0,0,0,0.15);
    --shadow-lv3: 0px 10px 20px 0px rgba(0,0,0,0.1), 0px 0px 1px 0px rgba(0,0,0,0.15);
    --shadow-brand: 0px 6px 10px 0px rgba(0,102,255,0.1), 0px 0px 1px 0px rgba(0,102,255,0.15);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
    --transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
}

* {
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Helvetica, 'Noto Sans', sans-serif;
}

.gradio-container {
    background: linear-gradient(109deg, #ffffff 45.34%, #f3f7ff 102.43%) !important;
    min-height: 100vh;
}

.main-container {
    background: var(--bg-primary) !important;
    border-radius: var(--radius-xl) !important;
    box-shadow: var(--shadow-lv3) !important;
    padding: 32px !important;
    margin: 24px auto !important;
    max-width: 1200px !important;
    border: 1px solid var(--border-tertiary) !important;
}

.title-area {
    text-align: center;
    margin-bottom: 24px;
}

.title-area h1 {
    color: var(--text-primary) !important;
    font-size: 26px !important;
    font-weight: 600 !important;
    line-height: 36px !important;
    margin-bottom: 8px !important;
}

.title-area p {
    color: var(--text-tertiary) !important;
    font-size: 16px !important;
    line-height: 24px !important;
}

.chatbot-container {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border-tertiary) !important;
    overflow: hidden !important;
    background: var(--bg-primary) !important;
    box-shadow: var(--shadow-lv1) !important;
}

.input-area textarea {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-secondary) !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    transition: var(--transition) !important;
}

.input-area textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0px 0px 0px 3px var(--primary-transparent-2) !important;
    outline: none !important;
}

.input-area textarea::placeholder {
    color: var(--text-tertiary) !important;
}

.send-btn {
    background: radial-gradient(125.81% 83.73% at 21.13% 50%, #0057ff 0%, #5a91fc 40.94%) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 500 !important;
    color: #ffffff !important;
    transition: var(--transition) !important;
    padding: 10px 20px !important;
}

.send-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-brand) !important;
}

.send-btn:active {
    transform: translateY(0) !important;
}

.clear-btn {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-secondary) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    transition: var(--transition) !important;
    font-weight: 400 !important;
}

.clear-btn:hover {
    background: var(--bg-tertiary) !important;
    border-color: var(--border-primary) !important;
}

.rebuild-btn {
    background: radial-gradient(125.81% 83.73% at 21.13% 50%, #0057ff 0%, #5a91fc 40.94%) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 500 !important;
    color: #ffffff !important;
    transition: var(--transition) !important;
}

.rebuild-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-brand) !important;
}

.health-btn {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-tertiary) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    transition: var(--transition) !important;
}

.health-btn:hover {
    background: var(--bg-tertiary) !important;
    border-color: var(--border-secondary) !important;
}

.sidebar-card {
    background: var(--bg-secondary) !important;
    border-radius: var(--radius-lg) !important;
    padding: 20px !important;
    border: 1px solid var(--border-tertiary) !important;
}

.sidebar-card h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    margin-bottom: 12px !important;
}

.sidebar-card p, .sidebar-card li {
    color: var(--text-secondary) !important;
    font-size: 14px !important;
    line-height: 22px !important;
}

.sidebar-card ul {
    padding-left: 20px !important;
}

.sidebar-card li {
    margin-bottom: 8px !important;
}

.status-box {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-tertiary) !important;
    background: var(--bg-primary) !important;
    color: var(--text-secondary) !important;
    font-size: 14px !important;
}

.message.user {
    background: radial-gradient(125.81% 83.73% at 21.13% 50%, #0057ff 0%, #5a91fc 40.94%) !important;
    color: #ffffff !important;
    border-radius: 18px 18px 5px 18px !important;
    padding: 12px 16px !important;
}

.message.assistant {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border-radius: 18px 18px 18px 5px !important;
    border: 1px solid var(--border-tertiary) !important;
    padding: 12px 16px !important;
}

.chatbot .message-wrap {
    gap: 16px !important;
}

.chatbot .user-message {
    background: radial-gradient(125.81% 83.73% at 21.13% 50%, #0057ff 0%, #5a91fc 40.94%) !important;
    color: #ffffff !important;
    border-radius: 18px 18px 5px 18px !important;
}

.chatbot .bot-message {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border-radius: 18px 18px 18px 5px !important;
    border: 1px solid var(--border-tertiary) !important;
}

button.primary {
    background: radial-gradient(125.81% 83.73% at 21.13% 50%, #0057ff 0%, #5a91fc 40.94%) !important;
}

button.secondary {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-secondary) !important;
    color: var(--text-primary) !important;
}

.gr-button {
    transition: var(--transition) !important;
}

.gr-button:hover {
    transform: translateY(-1px) !important;
}

footer {
    display: none !important;
}

.contain {
    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
}

.gap {
    gap: 16px !important;
}

.form {
    border: none !important;
    background: transparent !important;
}

.panel {
    border: 1px solid var(--border-tertiary) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--bg-primary) !important;
}
"""

with gr.Blocks(title="课程助手") as demo:
    with gr.Column(elem_classes=["main-container"]):
        with gr.Column(elem_classes=["title-area"]):
            gr.Markdown("""
            # 📚 智能课程助手
            基于 RAG 技术的智能问答系统，帮助您快速查询课程资料中的内容
            """)
        
        with gr.Row():
            health_btn = gr.Button("🔍 检查服务状态", elem_classes=["health-btn"], size="sm")
            health_status = gr.Textbox(label="", interactive=False, elem_classes=["status-box"], show_label=False)
        
        health_btn.click(check_health, outputs=health_status)
        
        with gr.Row():
            with gr.Column(scale=4, elem_classes=["chat-area"]):
                chatbot = gr.Chatbot(
                    label="",
                    height=500,
                    elem_classes=["chatbot-container"]
                )
                
                with gr.Row(elem_classes=["input-area"]):
                    question_input = gr.Textbox(
                        label="",
                        placeholder="💬 请输入您的问题，例如：多AGV调度系统研究的核心算法是什么？",
                        scale=4,
                        show_label=False,
                        container=False
                    )
                    submit_btn = gr.Button("发送 ➤", variant="primary", elem_classes=["send-btn"], scale=1)
                
                clear_btn = gr.Button("🗑️ 清空对话", elem_classes=["clear-btn"], size="sm")
            
            with gr.Column(scale=1, elem_classes=["sidebar"]):
                with gr.Column(elem_classes=["sidebar-card"]):
                    gr.Markdown("""
                    ### ⚙️ 管理操作
                    """)
                    rebuild_btn = gr.Button("🔄 重建知识库", elem_classes=["rebuild-btn"])
                    rebuild_status = gr.Textbox(label="", interactive=False, elem_classes=["status-box"], show_label=False)
                
                with gr.Column(elem_classes=["sidebar-card"]):
                    gr.Markdown("""
                    ### 📖 使用说明
                    
                    1. 在下方输入框输入问题
                    2. 点击"发送"或按回车键
                    3. 系统将从课程资料中检索相关内容
                    4. 如答案不准确，可尝试重建知识库
                    
                    ---
                    
                    ### 💡 提示
                    
                    - 问题越具体，回答越准确
                    - 重建知识库需要较长时间
                    - 支持中英文问答
                    """)
        
        submit_btn.click(
            ask_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input]
        )
        
        question_input.submit(
            ask_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input]
        )
        
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, question_input])
        
        rebuild_btn.click(rebuild_knowledge_base, outputs=rebuild_status)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=custom_css
    )
