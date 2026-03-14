const API_URL = 'http://127.0.0.1:8001';

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
    chatContainer: document.getElementById('chatContainer')
};

let isProcessing = false;

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
    if (indicator) {
        indicator.remove();
    }
}

function removeWelcomeMessage() {
    const welcome = elements.messages.querySelector('.welcome-message');
    if (welcome) {
        welcome.remove();
    }
}

function clearMessages() {
    elements.messages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">💬</div>
            <p>您好！我是智能课程助手，请输入您的问题开始对话。</p>
        </div>
    `;
}

async function checkHealth() {
    elements.healthStatus.textContent = '检查中...';
    elements.healthStatus.className = 'status-box';
    
    try {
        const response = await fetch(`${API_URL}/health`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
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
        const response = await fetch(`${API_URL}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        });
        
        removeTypingIndicator();
        
        if (response.ok) {
            const result = await response.json();
            const answer = result.answer || '抱歉，无法获取回答';
            const assistantMessage = createMessageElement(answer, 'assistant');
            elements.messages.appendChild(assistantMessage);
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

async function rebuildKnowledgeBase() {
    elements.rebuildStatus.textContent = '重建中...';
    elements.rebuildStatus.className = 'status-box';
    elements.rebuildBtn.disabled = true;
    
    showLoading();
    
    try {
        const response = await fetch(`${API_URL}/init?force_rebuild=true`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
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
        
        let errorMessage = '发生错误: ';
        if (error.message.includes('Failed to fetch')) {
            errorMessage += '无法连接到后端服务';
        } else {
            errorMessage += error.message;
        }
        
        elements.rebuildStatus.textContent = errorMessage;
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

elements.sendBtn.addEventListener('click', () => {
    askQuestion(elements.questionInput.value);
});

elements.questionInput.addEventListener('keydown', handleKeyDown);

elements.questionInput.addEventListener('input', autoResizeTextarea);

elements.clearBtn.addEventListener('click', clearMessages);

elements.healthBtn.addEventListener('click', checkHealth);

elements.rebuildBtn.addEventListener('click', rebuildKnowledgeBase);

document.addEventListener('DOMContentLoaded', () => {
    elements.questionInput.focus();
    checkHealth();
});
