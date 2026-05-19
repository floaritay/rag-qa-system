const API_URL = 'http://127.0.0.1:8001';

// ============================================================
// 状态
// ============================================================
let isProcessing = false;
let currentSessionId = null;

// ============================================================
// DOM 元素
// ============================================================
const elements = {
    messages: document.getElementById('messages'),
    questionInput: document.getElementById('questionInput'),
    sendBtn: document.getElementById('sendBtn'),
    clearBtn: document.getElementById('clearBtn'),
    healthBtn: document.getElementById('healthBtn'),
    healthStatus: document.getElementById('healthStatus'),
    rebuildBtn: document.getElementById('rebuildBtn'),
    rebuildStatus: document.getElementById('rebuildStatus'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    chatContainer: document.getElementById('chatContainer'),
    newSessionBtn: document.getElementById('newSessionBtn'),
    sessionList: document.getElementById('sessionList'),
    sessionBanner: document.getElementById('sessionBanner'),
    sessionBannerTitle: document.getElementById('sessionBannerTitle'),
    closeSessionBtn: document.getElementById('closeSessionBtn')
};

// ============================================================
// 工具函数
// ============================================================

function showLoading() {
    elements.loadingOverlay.classList.add('active');
}

function hideLoading() {
    elements.loadingOverlay.classList.remove('active');
}

function scrollToBottom() {
    elements.messages.scrollTop = elements.messages.scrollHeight;
}

function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

// ============================================================
// 消息渲染
// ============================================================

function createMessageElement(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = type === 'user' ? '👤' : '🤖';

    const contentWrapper = document.createElement('div');
    contentWrapper.style.display = 'flex';
    contentWrapper.style.flexDirection = 'column';
    contentWrapper.style.maxWidth = '70%';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    const timeDiv = document.createElement('span');
    timeDiv.className = 'message-time';
    timeDiv.textContent = getCurrentTime();

    contentWrapper.appendChild(contentDiv);
    contentWrapper.appendChild(timeDiv);

    if (type === 'user') {
        messageDiv.appendChild(contentWrapper);
        messageDiv.appendChild(avatarDiv);
    } else {
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentWrapper);
    }

    return messageDiv;
}

function createTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = 'typingIndicator';

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = '🤖';

    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';

    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(typingDiv);

    return messageDiv;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) indicator.remove();
}

function removeWelcomeMessage() {
    const welcome = elements.messages.querySelector('.welcome-message');
    if (welcome) welcome.remove();
}

function clearMessages() {
    currentSessionId = null;
    updateSessionBanner();
    elements.messages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">💬</div>
            <p>您好！我是智能课程助手，请输入您的问题开始对话。</p>
        </div>
    `;
    loadSessionList();
}

// ============================================================
// 会话管理
// ============================================================

async function loadSessionList() {
    try {
        const response = await fetch(`${API_URL}/sessions`);
        if (!response.ok) return;
        const data = await response.json();
        renderSessionList(data.sessions || []);
    } catch (e) {
        // 静默失败
    }
}

function renderSessionList(sessions) {
    elements.sessionList.innerHTML = '';
    if (sessions.length === 0) {
        elements.sessionList.innerHTML = `
            <div class="session-list-empty">
                <div class="empty-icon">📭</div>
                <p>暂无会话</p>
                <p>点击上方"＋"新建</p>
            </div>
        `;
        return;
    }
    sessions.forEach(s => {
        const item = document.createElement('div');
        item.className = 'session-item' + (s.session_id === currentSessionId ? ' active' : '');
        item.innerHTML = `
            <div class="session-item-title">${escapeHtml(s.title)}</div>
            <div class="session-item-meta">
                <span>${s.message_count} 条消息</span>
                <span>${formatTime(s.updated_at)}</span>
            </div>
            <button class="session-item-delete" data-id="${s.session_id}" title="删除会话">✕</button>
        `;
        item.addEventListener('click', (e) => {
            if (e.target.classList.contains('session-item-delete')) return;
            switchSession(s.session_id);
        });
        item.querySelector('.session-item-delete').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteSession(s.session_id);
        });
        elements.sessionList.appendChild(item);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(isoStr) {
    if (!isoStr) return '';
    const d = new Date(isoStr);
    const now = new Date();
    const diffMs = now - d;
    const diffMin = Math.floor(diffMs / 60000);
    if (diffMin < 1) return '刚刚';
    if (diffMin < 60) return `${diffMin}分钟前`;
    const diffHour = Math.floor(diffMin / 60);
    if (diffHour < 24) return `${diffHour}小时前`;
    return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
}

async function createNewSession() {
    try {
        const response = await fetch(`${API_URL}/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: '新对话' })
        });
        if (!response.ok) return;
        const session = await response.json();
        switchSession(session.session_id);
    } catch (e) {
        console.error('创建会话失败:', e);
    }
}

async function switchSession(sessionId) {
    currentSessionId = sessionId;
    updateSessionBanner();
    elements.messages.innerHTML = '';
    removeWelcomeMessage();

    try {
        const response = await fetch(`${API_URL}/sessions/${sessionId}/messages`);
        if (response.ok) {
            const data = await response.json();
            const messages = data.messages || [];
            if (messages.length === 0) {
                elements.messages.innerHTML = `
                    <div class="welcome-message">
                        <div class="welcome-icon">💬</div>
                        <p>会话已开始，请输入您的问题。</p>
                    </div>
                `;
            } else {
                messages.forEach(msg => {
                    const type = msg.role === 'user' ? 'user' : 'assistant';
                    elements.messages.appendChild(createMessageElement(msg.content, type));
                });
                scrollToBottom();
            }
        }
        // 更新标题
        const sessionRes = await fetch(`${API_URL}/sessions/${sessionId}`);
        if (sessionRes.ok) {
            const session = await sessionRes.json();
            elements.sessionBannerTitle.textContent = session.title || '新对话';
        }
    } catch (e) {
        console.error('加载会话历史失败:', e);
    }

    loadSessionList();
}

async function deleteSession(sessionId) {
    try {
        await fetch(`${API_URL}/sessions/${sessionId}`, { method: 'DELETE' });
        if (currentSessionId === sessionId) {
            clearMessages();
        }
        loadSessionList();
    } catch (e) {
        console.error('删除会话失败:', e);
    }
}

function updateSessionBanner() {
    if (currentSessionId) {
        elements.sessionBanner.style.display = 'flex';
    } else {
        elements.sessionBanner.style.display = 'none';
    }
}

function exitSession() {
    currentSessionId = null;
    updateSessionBanner();
    clearMessages();
}

// ============================================================
// 问答
// ============================================================

async function askQuestion(question) {
    if (!question.trim() || isProcessing) return;

    isProcessing = true;
    elements.sendBtn.disabled = true;

    removeWelcomeMessage();

    const userMessage = createMessageElement(question, 'user');
    elements.messages.appendChild(userMessage);
    scrollToBottom();

    elements.questionInput.value = '';
    autoResizeTextarea();

    const typingIndicator = createTypingIndicator();
    elements.messages.appendChild(typingIndicator);
    scrollToBottom();

    try {
        // 如果没有当前会话，自动创建一个
        if (!currentSessionId) {
            try {
                const sessRes = await fetch(`${API_URL}/sessions`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: '新对话' })
                });
                if (sessRes.ok) {
                    const sess = await sessRes.json();
                    currentSessionId = sess.session_id;
                    updateSessionBanner();
                    elements.sessionBannerTitle.textContent = '新对话';
                }
            } catch (e) {
                console.error('自动创建会话失败:', e);
            }
        }

        const body = { question: question };
        if (currentSessionId) {
            body.session_id = currentSessionId;
        }

        const response = await fetch(`${API_URL}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        removeTypingIndicator();

        if (response.ok) {
            const result = await response.json();
            const answer = result.answer || '抱歉，无法获取回答';
            const assistantMessage = createMessageElement(answer, 'assistant');
            elements.messages.appendChild(assistantMessage);

            // 如果是新会话且后端返回了 session_id，更新本地状态
            if (result.session_id && !currentSessionId) {
                currentSessionId = result.session_id;
                updateSessionBanner();
            }
            // 刷新会话列表（标题可能已更新）
            loadSessionList();
        } else {
            const errorData = await response.json();
            const errorMessage = createMessageElement(
                `错误: ${errorData.detail || '未知错误'}`,
                'error'
            );
            elements.messages.appendChild(errorMessage);
        }
    } catch (error) {
        removeTypingIndicator();

        let errorMessage = '发生错误: ';
        if (error.name === 'AbortError') {
            errorMessage += '请求超时，请稍后重试';
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage += '无法连接到后端服务，请确保后端已启动';
        } else {
            errorMessage += error.message;
        }

        const errorElement = createMessageElement(errorMessage, 'error');
        elements.messages.appendChild(errorElement);
    }

    scrollToBottom();
    isProcessing = false;
    elements.sendBtn.disabled = false;
    elements.questionInput.focus();
}

// ============================================================
// 其他功能
// ============================================================

async function checkHealth() {
    elements.healthStatus.textContent = '检查中...';
    elements.healthStatus.className = 'status-box';

    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            elements.healthStatus.textContent = '✅ 后端服务正常运行';
            elements.healthStatus.className = 'status-box success';
        } else {
            elements.healthStatus.textContent = '❌ 后端服务异常';
            elements.healthStatus.className = 'status-box error';
        }
    } catch (error) {
        elements.healthStatus.textContent = '❌ 无法连接到后端服务';
        elements.healthStatus.className = 'status-box error';
    }
}

async function rebuildKnowledgeBase() {
    elements.rebuildStatus.textContent = '重建中...';
    elements.rebuildStatus.className = 'status-box';
    elements.rebuildBtn.disabled = true;

    showLoading();

    try {
        const response = await fetch(`${API_URL}/init?force_rebuild=true`, { method: 'POST' });
        hideLoading();

        if (response.ok) {
            const result = await response.json();
            elements.rebuildStatus.textContent = result.message || '知识库重建成功';
            elements.rebuildStatus.className = 'status-box success';
        } else {
            const errorData = await response.json();
            elements.rebuildStatus.textContent = `错误: ${errorData.detail || '未知错误'}`;
            elements.rebuildStatus.className = 'status-box error';
        }
    } catch (error) {
        hideLoading();
        elements.rebuildStatus.textContent = '无法连接到后端服务';
        elements.rebuildStatus.className = 'status-box error';
    }

    elements.rebuildBtn.disabled = false;
}

function autoResizeTextarea() {
    const textarea = elements.questionInput;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        askQuestion(elements.questionInput.value);
    }
}

// ============================================================
// 事件绑定
// ============================================================

elements.sendBtn.addEventListener('click', () => askQuestion(elements.questionInput.value));
elements.questionInput.addEventListener('keydown', handleKeyDown);
elements.questionInput.addEventListener('input', autoResizeTextarea);
elements.clearBtn.addEventListener('click', clearMessages);
elements.healthBtn.addEventListener('click', checkHealth);
elements.rebuildBtn.addEventListener('click', rebuildKnowledgeBase);
elements.newSessionBtn.addEventListener('click', createNewSession);
elements.closeSessionBtn.addEventListener('click', exitSession);

document.addEventListener('DOMContentLoaded', () => {
    elements.questionInput.focus();
    checkHealth();
    loadSessionList();
});
