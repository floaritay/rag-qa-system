const API_URL = 'http://127.0.0.1:8001';

// ============================================================
// State
// ============================================================
let isProcessing = false;
let currentSessionId = null;
let currentKbId = 'default';

// ============================================================
// DOM
// ============================================================
const $ = (id) => document.getElementById(id);

const dom = {
    sidebar: $('sidebar'),
    menuBtn: $('menuBtn'),
    sidebarOverlay: $('sidebarOverlay'),
    newChatBtn: $('newChatBtn'),
    sessionList: $('sessionList'),
    chatMessages: $('chatMessages'),
    welcome: $('welcome'),
    topbarTitle: $('topbarTitle'),
    healthIndicator: $('healthIndicator'),
    questionInput: $('questionInput'),
    sendBtn: $('sendBtn'),
    settingsBtn: $('settingsBtn'),
    settingsOverlay: $('settingsOverlay'),
    settingsClose: $('settingsClose'),
    modelSettingsBtn: $('modelSettingsBtn'),
    modelSettingsOverlay: $('modelSettingsOverlay'),
    modelSettingsClose: $('modelSettingsClose'),
    retrievalStrategySelect: $('retrievalStrategySelect'),
    preRetrievalSelect: $('preRetrievalSelect'),
    postRetrievalSelect: $('postRetrievalSelect'),
    strategyConflict: $('strategyConflict'),
    strategyConflictMsg: $('strategyConflictMsg'),
    toastContainer: $('toastContainer'),
    cfgLlmModel: $('cfgLlmModel'),
    cfgLlmBaseUrl: $('cfgLlmBaseUrl'),
    cfgLlmApiKey: $('cfgLlmApiKey'),
    cfgEmbModel: $('cfgEmbModel'),
    cfgEmbBaseUrl: $('cfgEmbBaseUrl'),
    cfgEmbApiKey: $('cfgEmbApiKey'),
    cfgRerankerModel: $('cfgRerankerModel'),
    cfgRerankerBaseUrl: $('cfgRerankerBaseUrl'),
    cfgRerankerApiKey: $('cfgRerankerApiKey'),
    saveConfigBtn: $('saveConfigBtn'),
    sfApiKey: $('sfApiKey'),
    sfLlmModelSelect: $('sfLlmModelSelect'),
    sfEmbModelSelect: $('sfEmbModelSelect'),
    sfRerankerModelSelect: $('sfRerankerModelSelect'),
    kbManageBtn: $('kbManageBtn'),
    kbOverlay: $('kbOverlay'),
    kbClose: $('kbClose'),
    kbFileList: $('kbFileList'),
    kbFileInput: $('kbFileInput'),
    kbUploadArea: $('kbUploadArea'),
    kbSelect: $('kbSelect'),
    kbSelectMenu: $('kbSelectMenu'),
    kbList: $('kbList'),
    kbNameInput: $('kbNameInput'),
    kbCreateBtn: $('kbCreateBtn'),
    kbFilesTitle: $('kbFilesTitle'),
};

// ============================================================
// Markdown Renderer (no external dependencies)
// ============================================================
function renderMarkdown(text) {
    try {
        let html = escapeHtml(text);

        // Code blocks ```...```
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            return `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`;
        });

        // Inline code `...`
        html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');

        // Tables
        html = html.replace(/^(\|.+\|)\n(\|[-| :]+\|)\n((?:\|.+\|\n?)*)/gm, (_, header, sep, body) => {
            const hCells = header.split('|').filter(c => c.trim()).map(c => `<th>${c.trim()}</th>`).join('');
            const rows = body.trim().split('\n').map(row => {
                const cells = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
                return `<tr>${cells}</tr>`;
            }).join('');
            return `<table><thead><tr>${hCells}</tr></thead><tbody>${rows}</tbody></table>`;
        });

        // Headings
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        // Blockquotes
        html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

        // Bold & italic
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

        // Unordered lists
        html = html.replace(/(?:^|\n)((?:[-*] .+\n?)+)/g, (match, list) => {
            const items = list.trim().split('\n').map(li => `<li>${li.replace(/^[-*] /, '')}</li>`).join('');
            return `\n<ul>${items}</ul>`;
        });

        // Ordered lists
        html = html.replace(/(?:^|\n)((?:\d+\. .+\n?)+)/g, (match, list) => {
            const items = list.trim().split('\n').map(li => `<li>${li.replace(/^\d+\. /, '')}</li>`).join('');
            return `\n<ol>${items}</ol>`;
        });

        // Paragraphs — wrap remaining loose lines
        html = html.replace(/^(?!<[a-z/])((?!^\s*$).+)$/gm, '<p>$1</p>');

        // Clean up empty paragraphs
        html = html.replace(/<p>\s*<\/p>/g, '');

        // Line breaks
        html = html.replace(/\n{2,}/g, '\n');

        return html;
    } catch {
        return `<p>${escapeHtml(text)}</p>`;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================
// Toast
// ============================================================
function showToast(message, type = 'info', persistent = false) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    if (persistent) {
        toast.dataset.persistent = 'true';
    }
    toast.textContent = message;
    dom.toastContainer.appendChild(toast);
    if (!persistent) {
        setTimeout(() => {
            toast.classList.add('removing');
            setTimeout(() => toast.remove(), 250);
        }, 3000);
    }
}

function removeLoadingToast() {
    dom.toastContainer.querySelectorAll('[data-persistent="true"]').forEach(el => {
        el.classList.add('removing');
        setTimeout(() => el.remove(), 250);
    });
}

// ============================================================
// Health Check
// ============================================================
async function checkHealth() {
    try {
        const res = await fetch(`${API_URL}/health`);
        if (res.ok) {
            dom.healthIndicator.className = 'topbar-status online';
            dom.healthIndicator.querySelector('.status-text').textContent = '在线';
        } else {
            throw new Error();
        }
    } catch {
        dom.healthIndicator.className = 'topbar-status offline';
        dom.healthIndicator.querySelector('.status-text').textContent = '离线';
    }
}

// ============================================================
// Sidebar
// ============================================================
function updateMenuArrow() {
    const isMobile = window.innerWidth <= 768;
    const sidebarHidden = isMobile
        ? !dom.sidebar.classList.contains('open')
        : dom.sidebar.classList.contains('collapsed');
    dom.menuBtn.classList.toggle('flipped', sidebarHidden);
}

function toggleSidebar() {
    const isMobile = window.innerWidth <= 768;
    if (isMobile) {
        dom.sidebar.classList.toggle('open');
        dom.sidebarOverlay.classList.toggle('open');
    } else {
        dom.sidebar.classList.toggle('collapsed');
    }
    updateMenuArrow();
}

function closeSidebar() {
    dom.sidebar.classList.remove('open', 'collapsed');
    dom.sidebarOverlay.classList.remove('open');
    updateMenuArrow();
}

// ============================================================
// Session Management
// ============================================================
async function loadSessionList() {
    try {
        const res = await fetch(`${API_URL}/sessions?kb_id=${encodeURIComponent(currentKbId)}`);
        if (!res.ok) return;
        const data = await res.json();
        renderSessionList(data.sessions || []);
    } catch {
        // silent
    }
}

function renderSessionList(sessions) {
    dom.sessionList.innerHTML = '';
    if (sessions.length === 0) {
        dom.sessionList.innerHTML = `
            <div class="session-list-empty">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
                <p>暂无对话</p>
                <p>点击「新建对话」开始</p>
            </div>
        `;
        return;
    }

    // Group: today / older
    const now = new Date();
    const todayStr = now.toDateString();
    const today = [];
    const older = [];

    sessions.forEach(s => {
        const d = new Date(s.updated_at);
        if (d.toDateString() === todayStr) {
            today.push(s);
        } else {
            older.push(s);
        }
    });

    function renderGroup(label, items) {
        if (items.length === 0) return;
        const groupLabel = document.createElement('div');
        groupLabel.className = 'session-group-label';
        groupLabel.textContent = label;
        dom.sessionList.appendChild(groupLabel);

        items.forEach(s => {
            const item = document.createElement('div');
            item.className = 'session-item' + (s.session_id === currentSessionId ? ' active' : '');
            item.innerHTML = `
                <div class="session-item-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                    </svg>
                </div>
                <div class="session-item-content">
                    <div class="session-item-title">${escapeHtml(s.title)}</div>
                    <div class="session-item-meta">${s.message_count} 条消息</div>
                </div>
                <button class="session-item-delete" data-id="${s.session_id}" title="删除">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            `;
            item.addEventListener('click', (e) => {
                if (e.target.closest('.session-item-delete')) return;
                switchSession(s.session_id);
            });
            item.querySelector('.session-item-delete').addEventListener('click', (e) => {
                e.stopPropagation();
                deleteSession(s.session_id);
            });
            dom.sessionList.appendChild(item);
        });
    }

    renderGroup('今天', today);
    renderGroup('更早', older);
}

async function createNewSession() {
    try {
        const res = await fetch(`${API_URL}/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: '新对话', kb_id: currentKbId }),
        });
        if (!res.ok) return;
        const session = await res.json();
        switchSession(session.session_id);
        closeSidebar();
    } catch (e) {
        showToast('创建会话失败', 'error');
    }
}

async function switchSession(sessionId) {
    currentSessionId = sessionId;
    dom.chatMessages.innerHTML = '';

    try {
        const [msgRes, sessRes] = await Promise.all([
            fetch(`${API_URL}/sessions/${sessionId}/messages`),
            fetch(`${API_URL}/sessions/${sessionId}`),
        ]);

        if (msgRes.ok) {
            const data = await msgRes.json();
            const messages = data.messages || [];
            if (messages.length === 0) {
                showWelcome();
            } else {
                messages.forEach(msg => {
                    addMessage(msg.content, msg.role === 'user' ? 'user' : 'assistant', false);
                });
                scrollChatBottom();
            }
        }

        if (sessRes.ok) {
            const session = await sessRes.json();
            dom.topbarTitle.textContent = session.title || '新对话';
        }
    } catch (e) {
        showToast('加载会话失败', 'error');
    }

    loadSessionList();
}

async function deleteSession(sessionId) {
    try {
        await fetch(`${API_URL}/sessions/${sessionId}`, { method: 'DELETE' });
        if (currentSessionId === sessionId) {
            currentSessionId = null;
            dom.topbarTitle.textContent = '个人知识库';
            dom.chatMessages.innerHTML = '';
            showWelcome();
        }
        loadSessionList();
    } catch (e) {
        showToast('删除会话失败', 'error');
    }
}

// ============================================================
// Welcome Screen
// ============================================================
function bindChipClicks() {
    document.querySelectorAll('.chip[data-question]').forEach(chip => {
        chip.addEventListener('click', () => {
            const q = chip.getAttribute('data-question');
            dom.questionInput.value = q;
            askQuestion(q);
        });
    });
}

// ============================================================
// Messages
// ============================================================
function addMessage(content, type, animate = true, sources = null) {
    // Remove welcome if present
    const welcome = dom.chatMessages.querySelector('.welcome');
    if (welcome) welcome.remove();

    const msg = document.createElement('div');
    msg.className = `message ${type}`;
    if (!animate) msg.style.animation = 'none';

    const avatarLabel = type === 'user' ? '你' : 'AI';
    const roleLabel = type === 'user' ? 'You' : 'Assistant';

    const renderedContent = type === 'assistant' ? renderMarkdown(content) : `<p>${escapeHtml(content)}</p>`;

    let sourcesHtml = '';
    if (type === 'assistant' && sources && sources.length > 0) {
        const items = sources.map(s => {
            const fileName = s.source ? s.source.split(/[\\/]/).pop() : '未知';
            const pageLabel = s.page != null ? ` · 第 ${s.page + 1} 页` : '';

            const badges = [];
            if (s.vector_score != null) {
                const label = s.vector_score < 1 ? '优' : s.vector_score < 2 ? '良' : '一般';
                badges.push(`<span class="source-badge score-vector" title="FAISS L2 距离，越小越相关">距离 ${s.vector_score.toFixed(2)}</span>`);
            }
            if (s.rrf_score != null) {
                badges.push(`<span class="source-badge score-rrf" title="RRF 融合分数，越大越相关">RRF ${s.rrf_score.toFixed(4)}</span>`);
            }
            if (s.rerank_score != null) {
                badges.push(`<span class="source-badge score-rerank" title="重排序相关度，越大越相关">${(s.rerank_score * 100).toFixed(1)}%</span>`);
            }

            return `
                <div class="source-item">
                    <div class="source-header">
                        <span class="source-rank">#${s.rank}</span>
                        <span class="source-file">${escapeHtml(fileName)}${pageLabel}</span>
                        <span class="source-badges">${badges.join('')}</span>
                    </div>
                    <div class="source-text">${escapeHtml(s.content)}</div>
                </div>
            `;
        }).join('');

        sourcesHtml = `
            <div class="sources-panel">
                <button class="sources-toggle" type="button">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="6 9 12 15 18 9"/>
                    </svg>
                    <span>检索来源（${sources.length} 条）</span>
                </button>
                <div class="sources-list">${items}</div>
            </div>
        `;
    }

    msg.innerHTML = `
        <div class="message-avatar">${avatarLabel}</div>
        <div class="message-body">
            <div class="message-role">${roleLabel}</div>
            <div class="message-content">${renderedContent}</div>
            ${sourcesHtml}
        </div>
    `;

    // Bind sources toggle
    if (sourcesHtml) {
        const toggle = msg.querySelector('.sources-toggle');
        const list = msg.querySelector('.sources-list');
        toggle.addEventListener('click', () => {
            toggle.classList.toggle('open');
            list.classList.toggle('open');
        });
    }

    dom.chatMessages.appendChild(msg);
    return msg;
}

function createStreamingMessage() {
    const welcome = dom.chatMessages.querySelector('.welcome');
    if (welcome) welcome.remove();

    const msg = document.createElement('div');
    msg.className = 'message assistant';
    msg.innerHTML = `
        <div class="message-avatar" style="background: linear-gradient(135deg, #c87941 0%, #a85d30 100%); color: #fff; flex-shrink: 0; width: 32px; height: 32px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 14px;">AI</div>
        <div class="message-body">
            <div class="message-role">Assistant</div>
            <div class="message-content"><span class="streaming-cursor"></span></div>
            <div class="sources-panel-slot"></div>
        </div>
    `;
    dom.chatMessages.appendChild(msg);
    scrollChatBottom();
    return msg;
}

function renderSourcesPanel(container, sources) {
    if (!sources || sources.length === 0) return;

    const items = sources.map(s => {
        const fileName = s.source ? s.source.split(/[\\/]/).pop() : '未知';
        const pageLabel = s.page != null ? ` · 第 ${s.page + 1} 页` : '';

        const badges = [];
        if (s.vector_score != null) {
            badges.push(`<span class="source-badge score-vector" title="FAISS L2 距离，越小越相关">距离 ${s.vector_score.toFixed(2)}</span>`);
        }
        if (s.rrf_score != null) {
            badges.push(`<span class="source-badge score-rrf" title="RRF 融合分数，越大越相关">RRF ${s.rrf_score.toFixed(4)}</span>`);
        }
        if (s.rerank_score != null) {
            badges.push(`<span class="source-badge score-rerank" title="重排序相关度，越大越相关">${(s.rerank_score * 100).toFixed(1)}%</span>`);
        }

        return `
            <div class="source-item">
                <div class="source-header">
                    <span class="source-rank">#${s.rank}</span>
                    <span class="source-file">${escapeHtml(fileName)}${pageLabel}</span>
                    <span class="source-badges">${badges.join('')}</span>
                </div>
                <div class="source-text">${escapeHtml(s.content)}</div>
            </div>
        `;
    }).join('');

    container.innerHTML = `
        <div class="sources-panel">
            <button class="sources-toggle" type="button">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="6 9 12 15 18 9"/>
                </svg>
                <span>检索来源（${sources.length} 条）</span>
            </button>
            <div class="sources-list">${items}</div>
        </div>
    `;

    const toggle = container.querySelector('.sources-toggle');
    const list = container.querySelector('.sources-list');
    toggle.addEventListener('click', () => {
        toggle.classList.toggle('open');
        list.style.display = list.style.display === 'block' ? 'none' : 'block';
    });
}

function addErrorMessage(content) {
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    msg.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-body">
            <div class="message-role">Assistant</div>
            <div class="message-content">${escapeHtml(content)}</div>
        </div>
    `;
    msg.querySelector('.message-content').classList.add('message-content');
    // Style as error via parent class
    msg.classList.add('error');
    dom.chatMessages.appendChild(msg);
}

function addTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.id = 'typingIndicator';
    indicator.innerHTML = `
        <div class="message-avatar" style="background: linear-gradient(135deg, #c87941 0%, #a85d30 100%); color: #fff; flex-shrink: 0; width: 32px; height: 32px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 14px;">AI</div>
        <div class="typing-dots">
            <span></span><span></span><span></span>
        </div>
    `;
    dom.chatMessages.appendChild(indicator);
    scrollChatBottom();
}

function removeTypingIndicator() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
}

function scrollChatBottom() {
    requestAnimationFrame(() => {
        dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    });
}

// ============================================================
// Ask Question
// ============================================================
async function askQuestion(question) {
    if (!question.trim() || isProcessing) return;

    isProcessing = true;
    dom.sendBtn.disabled = true;

    addMessage(question, 'user');
    dom.questionInput.value = '';
    autoResize();
    scrollChatBottom();

    addTypingIndicator();

    try {
        // Auto-create session if needed
        if (!currentSessionId) {
            try {
                const sessRes = await fetch(`${API_URL}/sessions`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: '新对话', kb_id: currentKbId }),
                });
                if (sessRes.ok) {
                    const sess = await sessRes.json();
                    currentSessionId = sess.session_id;
                    dom.topbarTitle.textContent = '新对话';
                }
            } catch {
                // continue without session
            }
        }

        const body = { question, kb_id: currentKbId };
        if (currentSessionId) body.session_id = currentSessionId;
        body.retrieval_strategy = getSelectValue('retrievalStrategySelect');
        body.pre_retrieval = getSelectValue('preRetrievalSelect');
        body.post_retrieval = getSelectValue('postRetrievalSelect');

        const res = await fetch(`${API_URL}/ask/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        removeTypingIndicator();

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            addErrorMessage(`请求失败：${err.detail || '未知错误'}`);
        } else {
            // Pre-create assistant message shell
            const msgEl = createStreamingMessage();
            const contentEl = msgEl.querySelector('.message-content');
            const sourcesContainer = msgEl.querySelector('.sources-panel-slot');
            let accumulated = '';
            let sourcesRendered = false;

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // keep incomplete line in buffer

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed.startsWith('data: ')) continue;
                    const data = trimmed.slice(6);
                    if (data === '[DONE]') continue;

                    try {
                        const event = JSON.parse(data);
                        if (event.type === 'sources' && !sourcesRendered) {
                            sourcesRendered = true;
                            renderSourcesPanel(sourcesContainer, event.data);
                        } else if (event.type === 'token') {
                            accumulated += event.data;
                            contentEl.innerHTML = renderMarkdown(accumulated);
                            scrollChatBottom();
                        } else if (event.type === 'error') {
                            accumulated = `[错误] ${event.data}`;
                            contentEl.innerHTML = `<p style="color: var(--red);">请求失败：${escapeHtml(event.data)}</p>`;
                        }
                    } catch {
                        // skip malformed JSON
                    }
                }
            }

            if (!accumulated) {
                contentEl.innerHTML = '<p>抱歉，无法获取回答</p>';
            }
            loadSessionList();
        }
    } catch (error) {
        removeTypingIndicator();
        if (error.message?.includes('Failed to fetch')) {
            addErrorMessage('无法连接到后端服务，请确认后端已启动');
        } else {
            addErrorMessage(`发生错误：${error.message}`);
        }
    }

    scrollChatBottom();
    isProcessing = false;
    dom.sendBtn.disabled = !dom.questionInput.value.trim();
    dom.questionInput.focus();
}

// ============================================================
// Rebuild Knowledge Base
// ============================================================
async function rebuildKnowledgeBase(kbId) {
    kbId = kbId || currentKbId;
    showToast('重建中...', 'info', true);
    try {
        const res = await fetch(`${API_URL}/knowledge-bases/${encodeURIComponent(kbId)}/rebuild`, { method: 'POST' });
        removeLoadingToast();
        if (res.ok) {
            const result = await res.json();
            showToast(result.message || '知识库重建成功', 'success');
        } else {
            const err = await res.json().catch(() => ({}));
            showToast(`重建失败：${err.detail || '未知错误'}`, 'error');
        }
    } catch {
        removeLoadingToast();
        showToast('无法连接到后端服务', 'error');
    }
}

// ============================================================
// Knowledge Base Management
// ============================================================
const FILE_ICONS = { '.pdf': '📄', '.pptx': '📊', '.docx': '📝', '.md': '📋' };

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function openKbPanel() {
    console.log('[KB] openKbPanel called');
    dom.kbOverlay.classList.add('open');
    loadKnowledgeBasesForPanel();
    loadMaterialFiles();
}

function closeKbPanel() {
    dom.kbOverlay.classList.remove('open');
}

// ============================================================
// Knowledge Base Management
// ============================================================
async function loadKnowledgeBases() {
    try {
        console.log('[KB] loadKnowledgeBases: fetching...');
        const res = await fetch(`${API_URL}/knowledge-bases`);
        console.log('[KB] loadKnowledgeBases: status', res.status);
        if (!res.ok) return;
        const data = await res.json();
        console.log('[KB] loadKnowledgeBases: data', data);
        renderKbSelector(data.knowledge_bases || []);
    } catch (e) {
        console.error('[KB] loadKnowledgeBases error:', e);
    }
}

function renderKbSelector(kbs) {
    dom.kbSelectMenu.innerHTML = '';
    kbs.forEach(kb => {
        const opt = document.createElement('div');
        opt.className = 'custom-select-option' + (kb.id === currentKbId ? ' selected' : '');
        opt.dataset.value = kb.id;
        opt.innerHTML = `<span class="option-label">${escapeHtml(kb.name)}</span>`;
        opt.addEventListener('click', (e) => {
            e.stopPropagation();
            dom.kbSelect.classList.remove('open');
            switchKnowledgeBase(kb.id);
        });
        dom.kbSelectMenu.appendChild(opt);
    });
    // Update trigger label
    const current = kbs.find(k => k.id === currentKbId);
    const label = dom.kbSelect.querySelector('.custom-select-label');
    if (label) label.textContent = current ? current.name : '默认知识库';
    dom.kbSelect.dataset.value = currentKbId;
}

async function switchKnowledgeBase(kbId) {
    currentKbId = kbId;
    currentSessionId = null;
    dom.kbSelect.dataset.value = kbId;
    // Close dropdown
    dom.kbSelect.classList.remove('open');
    // Reload sessions for this KB
    await loadSessionList();
    // Reload KB selector to update selected state
    await loadKnowledgeBases();
    // Show welcome screen
    showWelcome();
}

function showWelcome() {
    dom.chatMessages.innerHTML = '';
    const welcome = document.createElement('div');
    welcome.className = 'welcome';
    welcome.id = 'welcome';
    welcome.innerHTML = `
        <div class="welcome-glow"></div>
        <div class="welcome-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                <path d="M2 17l10 5 10-5"/>
                <path d="M2 12l10 5 10-5"/>
            </svg>
        </div>
        <h2>个人知识库</h2>
        <p>基于 RAG 技术，精准检索文档，为您解答专业问题</p>
        <div class="welcome-chips">
            <button class="chip" data-question="这门课程的主要内容是什么？">课程主要内容</button>
            <button class="chip" data-question="请总结一下最近讲的知识点">知识点总结</button>
            <button class="chip" data-question="有哪些重要的概念需要掌握？">重要概念</button>
        </div>
    `;
    dom.chatMessages.appendChild(welcome);
    dom.topbarTitle.textContent = '个人知识库';
    // Re-bind chip clicks
    welcome.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => askQuestion(chip.dataset.question));
    });
}

async function loadKnowledgeBasesForPanel() {
    try {
        console.log('[KB] loadKnowledgeBasesForPanel: fetching...');
        const res = await fetch(`${API_URL}/knowledge-bases`);
        console.log('[KB] loadKnowledgeBasesForPanel: status', res.status);
        if (!res.ok) return;
        const data = await res.json();
        console.log('[KB] loadKnowledgeBasesForPanel: data', data);
        renderKbList(data.knowledge_bases || []);
    } catch (e) {
        console.error('[KB] loadKnowledgeBasesForPanel error:', e);
    }
}

function renderKbList(kbs) {
    console.log('[KB] renderKbList:', kbs.length, 'items');
    dom.kbList.innerHTML = '';
    kbs.forEach(kb => {
        const item = document.createElement('div');
        item.className = 'kb-list-item' + (kb.id === currentKbId ? ' active' : '');
        item.dataset.kbId = kb.id;

        const info = document.createElement('div');
        info.className = 'kb-list-item-info';
        info.innerHTML = `
            <div class="kb-list-item-name">${escapeHtml(kb.name)}</div>
            <div class="kb-list-item-meta">${kb.file_count} 个文档</div>
        `;

        const actions = document.createElement('div');
        actions.className = 'kb-actions';

        const rebuildBtn = document.createElement('button');
        rebuildBtn.className = 'btn-icon';
        rebuildBtn.title = '重建索引';
        rebuildBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>';
        rebuildBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            rebuildKnowledgeBase(kb.id);
        });

        actions.appendChild(rebuildBtn);

        if (kb.id !== 'default') {
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'btn-icon';
            deleteBtn.title = '删除知识库';
            deleteBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>';
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteKnowledgeBase(kb.id, kb.name);
            });
            actions.appendChild(deleteBtn);
        }

        item.append(info, actions);
        item.addEventListener('click', () => selectKbInPanel(kb.id));
        dom.kbList.appendChild(item);
    });
}

function selectKbInPanel(kbId) {
    currentKbId = kbId;
    currentSessionId = null;
    // Update active state in list
    dom.kbList.querySelectorAll('.kb-list-item').forEach(el => el.classList.remove('active'));
    const activeItem = dom.kbList.querySelector(`[data-kb-id="${kbId}"]`);
    if (activeItem) activeItem.classList.add('active');
    // Update sidebar selector label
    const label = dom.kbSelect.querySelector('.custom-select-label');
    const name = activeItem ? activeItem.querySelector('.kb-list-item-name')?.textContent : kbId;
    if (label && name) label.textContent = name;
    dom.kbSelect.dataset.value = kbId;
    // Reload sessions and files
    loadSessionList();
    loadMaterialFiles();
    showWelcome();
}

async function createKnowledgeBase() {
    const name = dom.kbNameInput.value.trim();
    if (!name) {
        showToast('请输入知识库名称', 'error');
        return;
    }
    try {
        const res = await fetch(`${API_URL}/knowledge-bases`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
        if (res.ok) {
            const kb = await res.json();
            dom.kbNameInput.value = '';
            showToast(`知识库 "${name}" 创建成功`, 'success');
            currentKbId = kb.id;
            await loadKnowledgeBases();
            await loadKnowledgeBasesForPanel();
            await loadMaterialFiles();
        } else {
            const err = await res.json().catch(() => ({}));
            showToast(`创建失败: ${err.detail || '未知错误'}`, 'error');
        }
    } catch {
        showToast('创建失败: 无法连接到后端', 'error');
    }
}

async function deleteKnowledgeBase(kbId, kbName) {
    if (!confirm(`确认删除知识库 "${kbName}"？此操作不可恢复。`)) return;
    try {
        const res = await fetch(`${API_URL}/knowledge-bases/${encodeURIComponent(kbId)}`, { method: 'DELETE' });
        if (res.ok) {
            showToast(`已删除: ${kbName}`, 'success');
            if (currentKbId === kbId) {
                currentKbId = 'default';
                currentSessionId = null;
                showWelcome();
            }
            await loadKnowledgeBases();
            await loadKnowledgeBasesForPanel();
            await loadSessionList();
        } else {
            const err = await res.json().catch(() => ({}));
            showToast(`删除失败: ${err.detail || '未知错误'}`, 'error');
        }
    } catch {
        showToast('删除失败: 无法连接到后端', 'error');
    }
}

async function loadMaterialFiles() {
    try {
        const res = await fetch(`${API_URL}/materials?kb_id=${encodeURIComponent(currentKbId)}`);
        if (res.ok) {
            const result = await res.json();
            renderFileList(result.files || []);
            // Update title
            if (dom.kbFilesTitle) {
                const label = dom.kbSelect.querySelector('.custom-select-label');
                const kbName = label ? label.textContent : currentKbId;
                dom.kbFilesTitle.textContent = `文档列表 — ${kbName}`;
            }
        } else {
            dom.kbFileList.innerHTML = '<div class="kb-empty">加载失败</div>';
        }
    } catch {
        dom.kbFileList.innerHTML = '<div class="kb-empty">无法连接到后端</div>';
    }
}

function renderFileList(files) {
    if (!files.length) {
        dom.kbFileList.innerHTML = '<div class="kb-empty">暂无文档</div>';
        return;
    }
    dom.kbFileList.innerHTML = '';
    files.forEach(f => {
        const ext = '.' + f.name.split('.').pop().toLowerCase();
        const icon = FILE_ICONS[ext] || '📄';
        const date = new Date(f.modified * 1000).toLocaleDateString('zh-CN');

        const item = document.createElement('div');
        item.className = 'kb-file-item';

        const iconSpan = document.createElement('span');
        iconSpan.className = 'kb-file-icon';
        iconSpan.textContent = icon;

        const info = document.createElement('div');
        info.className = 'kb-file-info';

        const nameDiv = document.createElement('div');
        nameDiv.className = 'kb-file-name';
        nameDiv.title = f.name;
        nameDiv.textContent = f.name;

        const metaDiv = document.createElement('div');
        metaDiv.className = 'kb-file-meta';
        metaDiv.textContent = `${formatFileSize(f.size)} · ${date}`;

        info.append(nameDiv, metaDiv);

        const delBtn = document.createElement('button');
        delBtn.className = 'kb-file-delete';
        delBtn.title = '删除';
        delBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>';
        delBtn.addEventListener('click', () => deleteMaterial(f.name));

        item.append(iconSpan, info, delBtn);
        dom.kbFileList.appendChild(item);
    });
}

async function uploadFiles(fileList) {
    if (!fileList.length) return;
    showToast('上传中...', 'info', true);
    const formData = new FormData();
    for (const file of fileList) formData.append('files', file);

    try {
        const res = await fetch(`${API_URL}/materials/upload?kb_id=${encodeURIComponent(currentKbId)}`, { method: 'POST', body: formData });
        const result = await res.json();
        removeLoadingToast();
        if (result.status === 'success') {
            showToast(`上传成功: ${result.uploaded.join(', ')}`, 'success');
        } else if (result.status === 'partial') {
            showToast(`部分上传成功: ${result.uploaded.join(', ')}`, 'info');
            if (result.errors) result.errors.forEach(e => showToast(e, 'error'));
        } else {
            showToast(`上传失败: ${(result.errors || [result.detail || '未知错误']).join(', ')}`, 'error');
        }
        loadMaterialFiles();
    } catch {
        removeLoadingToast();
        showToast('上传失败: 无法连接到后端', 'error');
    }
}

async function deleteMaterial(filename) {
    if (!confirm(`确认删除 "${filename}"？`)) return;
    try {
        const res = await fetch(`${API_URL}/materials/${encodeURIComponent(filename)}?kb_id=${encodeURIComponent(currentKbId)}`, { method: 'DELETE' });
        if (res.ok) {
            showToast(`已删除: ${filename}`, 'success');
            loadMaterialFiles();
        } else {
            const err = await res.json().catch(() => ({}));
            showToast(`删除失败: ${err.detail || '未知错误'}`, 'error');
        }
    } catch {
        showToast('删除失败: 无法连接到后端', 'error');
    }
}

// ============================================================
// Input Handling
// ============================================================
function autoResize() {
    const ta = dom.questionInput;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 150) + 'px';
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestion(dom.questionInput.value);
    }
}

// ============================================================
// Settings Panel & Custom Selects
// ============================================================
function openSettings() {
    dom.settingsOverlay.classList.add('open');
}

function closeSettings() {
    dom.settingsOverlay.classList.remove('open');
    document.querySelectorAll('.custom-select.open').forEach(s => s.classList.remove('open'));
}

function openModelSettings() {
    dom.modelSettingsOverlay.classList.add('open');
    loadConfig();
}

function closeModelSettings() {
    dom.modelSettingsOverlay.classList.remove('open');
}

function getSelectValue(id) {
    return document.getElementById(id).dataset.value;
}

function initCustomSelects() {
    document.querySelectorAll('.custom-select').forEach(select => {
        const trigger = select.querySelector('.custom-select-trigger');
        const menu = select.querySelector('.custom-select-menu');

        // Toggle dropdown
        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            const wasOpen = select.classList.contains('open');
            // Close all others
            document.querySelectorAll('.custom-select.open').forEach(s => s.classList.remove('open'));
            if (!wasOpen) select.classList.add('open');
        });

        // Option click
        menu.querySelectorAll('.custom-select-option').forEach(option => {
            option.addEventListener('click', (e) => {
                e.stopPropagation();
                const value = option.dataset.value;
                const label = option.querySelector('.option-label').textContent;

                // Update selection state
                menu.querySelectorAll('.custom-select-option').forEach(o => o.classList.remove('selected'));
                option.classList.add('selected');

                // Update trigger label & data
                select.dataset.value = value;
                select.querySelector('.custom-select-label').textContent = label;

                // Close dropdown
                select.classList.remove('open');

                // Check conflicts
                updateStrategyConstraints();
            });
        });
    });

    // Close all on outside click
    document.addEventListener('click', () => {
        document.querySelectorAll('.custom-select.open').forEach(s => s.classList.remove('open'));
    });
}

// ============================================================
// Per-Model Provider Mini Toggles
// ============================================================
const modelProviders = { llm: 'siliconflow', embedding: 'siliconflow', reranker: 'siliconflow' };

function initMiniToggles() {
    document.querySelectorAll('.mini-toggle').forEach(toggle => {
        const target = toggle.dataset.target;
        toggle.querySelectorAll('.mini-toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const provider = btn.dataset.value;
                modelProviders[target] = provider;
                toggle.querySelectorAll('.mini-toggle-btn').forEach(b => b.classList.toggle('active', b.dataset.value === provider));
                const section = toggle.closest('.settings-section-title').nextElementSibling;
                if (section) {
                    section.querySelector('.model-sf').style.display = provider === 'siliconflow' ? '' : 'none';
                    section.querySelector('.model-custom').style.display = provider === 'custom' ? '' : 'none';
                }
            });
        });
    });

    // Fetch models when API key changes (with debounce)
    let debounceTimer = null;
    dom.sfApiKey.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            const key = dom.sfApiKey.value.trim();
            if (key && !key.startsWith('***')) {
                fetchSiliconFlowModels(key);
            }
        }, 800);
    });
}

async function fetchSiliconFlowModels(apiKey) {
    const selects = [dom.sfLlmModelSelect, dom.sfEmbModelSelect, dom.sfRerankerModelSelect];
    selects.forEach(s => {
        const label = s.querySelector('.custom-select-label');
        label.textContent = '加载中...';
    });

    try {
        const url = `${API_URL}/models/siliconflow`;
        const headers = {};
        if (apiKey) headers['X-API-Key'] = apiKey;
        const res = await fetch(url, { headers });
        if (!res.ok) throw new Error('请求失败');
        const data = await res.json();
        const models = data.models || {};

        populateModelSelect(dom.sfLlmModelSelect, models.llm || []);
        populateModelSelect(dom.sfEmbModelSelect, models.embedding || []);
        populateModelSelect(dom.sfRerankerModelSelect, models.reranker || []);
    } catch {
        selects.forEach(s => {
            const label = s.querySelector('.custom-select-label');
            label.textContent = '加载失败，请重试';
        });
    }
}

function populateModelSelect(selectEl, models) {
    const menu = selectEl.querySelector('.custom-select-menu');
    const label = selectEl.querySelector('.custom-select-label');
    menu.innerHTML = '';

    if (models.length === 0) {
        label.textContent = '暂无可用模型';
        selectEl.dataset.value = '';
        return;
    }

    // Set first model as default
    selectEl.dataset.value = models[0];
    label.textContent = models[0];

    models.forEach(m => {
        const opt = document.createElement('div');
        opt.className = 'custom-select-option' + (m === models[0] ? ' selected' : '');
        opt.dataset.value = m;
        opt.innerHTML = `<span class="option-label">${escapeHtml(m)}</span>`;
        opt.addEventListener('click', (e) => {
            e.stopPropagation();
            menu.querySelectorAll('.custom-select-option').forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            selectEl.dataset.value = m;
            label.textContent = m;
            selectEl.classList.remove('open');
        });
        menu.appendChild(opt);
    });
}

function updateStrategyConstraints() {
    const pre = getSelectValue('preRetrievalSelect');
    const retrieval = getSelectValue('retrievalStrategySelect');
    const conflict = dom.strategyConflict;

    // Check HyDE + hybrid conflict
    if (pre === 'hyde' && retrieval === 'hybrid') {
        conflict.classList.add('visible');
    } else {
        conflict.classList.remove('visible');
    }
}

// ============================================================
// Config Load / Save (mixed providers)
// ============================================================
async function loadConfig() {
    try {
        const res = await fetch(`${API_URL}/config`);
        if (!res.ok) return;
        const cfg = await res.json();

        // SiliconFlow API key
        dom.sfApiKey.value = '';
        dom.sfApiKey.placeholder = cfg.siliconflow_api_key || '未设置';

        // Detect per-model provider from base_url
        const isSF = (url) => url && url.includes('siliconflow');
        const llmProv = isSF(cfg.llm?.base_url) ? 'siliconflow' : 'custom';
        const embProv = isSF(cfg.embedding?.base_url) ? 'siliconflow' : 'custom';
        const rerProv = isSF(cfg.reranker?.base_url) ? 'siliconflow' : 'custom';

        // Set mini toggles
        setMiniToggle('llm', llmProv);
        setMiniToggle('embedding', embProv);
        setMiniToggle('reranker', rerProv);

        // Fetch SiliconFlow models
        await fetchSiliconFlowModels();

        // Set SiliconFlow dropdown values
        if (cfg.llm?.model && llmProv === 'siliconflow') setCustomSelectValue(dom.sfLlmModelSelect, cfg.llm.model);
        if (cfg.embedding?.model && embProv === 'siliconflow') setCustomSelectValue(dom.sfEmbModelSelect, cfg.embedding.model);
        if (cfg.reranker?.model && rerProv === 'siliconflow') setCustomSelectValue(dom.sfRerankerModelSelect, cfg.reranker.model);

        // Custom fields — pre-fill with saved values when provider is custom
        if (llmProv === 'custom') {
            dom.cfgLlmModel.value = cfg.llm?.model || '';
            dom.cfgLlmBaseUrl.value = cfg.llm?.base_url || '';
            dom.cfgLlmApiKey.value = '';
            dom.cfgLlmApiKey.placeholder = cfg.llm?.api_key || '输入 API Key';
        } else {
            dom.cfgLlmModel.value = '';
            dom.cfgLlmBaseUrl.value = '';
            dom.cfgLlmApiKey.value = '';
            dom.cfgLlmBaseUrl.placeholder = '例如 https://api.openai.com/v1';
            dom.cfgLlmApiKey.placeholder = '输入 API Key';
        }

        if (embProv === 'custom') {
            dom.cfgEmbModel.value = cfg.embedding?.model || '';
            dom.cfgEmbBaseUrl.value = cfg.embedding?.base_url || '';
            dom.cfgEmbApiKey.value = '';
            dom.cfgEmbApiKey.placeholder = cfg.embedding?.api_key || '输入 API Key';
        } else {
            dom.cfgEmbModel.value = '';
            dom.cfgEmbBaseUrl.value = '';
            dom.cfgEmbApiKey.value = '';
            dom.cfgEmbBaseUrl.placeholder = '例如 https://api.openai.com/v1';
            dom.cfgEmbApiKey.placeholder = '输入 API Key';
        }

        if (rerProv === 'custom') {
            dom.cfgRerankerModel.value = cfg.reranker?.model || '';
            dom.cfgRerankerBaseUrl.value = cfg.reranker?.base_url || '';
            dom.cfgRerankerApiKey.value = '';
            dom.cfgRerankerApiKey.placeholder = cfg.reranker?.api_key || '输入 API Key';
        } else {
            dom.cfgRerankerModel.value = '';
            dom.cfgRerankerBaseUrl.value = '';
            dom.cfgRerankerApiKey.value = '';
            dom.cfgRerankerBaseUrl.placeholder = '例如 https://api.openai.com/v1';
            dom.cfgRerankerApiKey.placeholder = '输入 API Key';
        }
    } catch {
        // silent
    }
}

function setMiniToggle(target, provider) {
    modelProviders[target] = provider;
    const toggle = document.querySelector(`.mini-toggle[data-target="${target}"]`);
    if (!toggle) return;
    toggle.querySelectorAll('.mini-toggle-btn').forEach(b => b.classList.toggle('active', b.dataset.value === provider));
    const section = toggle.closest('.settings-section-title').nextElementSibling;
    if (section) {
        section.querySelector('.model-sf').style.display = provider === 'siliconflow' ? '' : 'none';
        section.querySelector('.model-custom').style.display = provider === 'custom' ? '' : 'none';
    }
}

function setCustomSelectValue(selectEl, value) {
    if (!value) return;
    const menu = selectEl.querySelector('.custom-select-menu');
    const label = selectEl.querySelector('.custom-select-label');
    selectEl.dataset.value = value;
    label.textContent = value;
    // Try to mark matching option as selected
    menu.querySelectorAll('.custom-select-option').forEach(o => {
        o.classList.toggle('selected', o.dataset.value === value);
    });
}

const SF_BASE = 'https://api.siliconflow.cn/v1';

async function saveConfig() {
    dom.saveConfigBtn.disabled = true;
    dom.saveConfigBtn.textContent = '保存中...';

    const sfKey = dom.sfApiKey.value.trim();

    // Build per-model configs
    const llmCfg = modelProviders.llm === 'siliconflow'
        ? { model: dom.sfLlmModelSelect.dataset.value, base_url: SF_BASE, api_key: sfKey }
        : { model: dom.cfgLlmModel.value.trim(), base_url: dom.cfgLlmBaseUrl.value.trim(), api_key: dom.cfgLlmApiKey.value.trim() };

    const embCfg = modelProviders.embedding === 'siliconflow'
        ? { model: dom.sfEmbModelSelect.dataset.value, base_url: SF_BASE, api_key: sfKey }
        : { model: dom.cfgEmbModel.value.trim(), base_url: dom.cfgEmbBaseUrl.value.trim(), api_key: dom.cfgEmbApiKey.value.trim() };

    const rerCfg = modelProviders.reranker === 'siliconflow'
        ? { model: dom.sfRerankerModelSelect.dataset.value, base_url: SF_BASE, api_key: sfKey }
        : { model: dom.cfgRerankerModel.value.trim(), base_url: dom.cfgRerankerBaseUrl.value.trim(), api_key: dom.cfgRerankerApiKey.value.trim() };

    // Send both old format (top-level llm/embedding/reranker) and new format (models)
    // for backward compatibility with older backend code
    const body = {
        siliconflow_api_key: sfKey,
        llm: llmCfg,
        embedding: embCfg,
        reranker: rerCfg,
        models: { llm: llmCfg, embedding: embCfg, reranker: rerCfg },
    };

    try {
        const res = await fetch(`${API_URL}/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const result = await res.json();
        if (res.ok) {
            showToast(result.message || '配置已保存', 'success');
            await loadConfig();
        } else {
            showToast(`保存失败：${result.detail || '未知错误'}`, 'error');
        }
    } catch {
        showToast('无法连接到后端服务', 'error');
    }

    dom.saveConfigBtn.disabled = false;
    dom.saveConfigBtn.textContent = '保存配置';
}

// Toggle API key visibility
document.querySelectorAll('.toggle-key-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const input = btn.parentElement.querySelector('input');
        input.type = input.type === 'password' ? 'text' : 'password';
    });
});

// ============================================================
// Event Binding
// ============================================================
dom.sendBtn.addEventListener('click', () => askQuestion(dom.questionInput.value));
dom.questionInput.addEventListener('keydown', handleKeyDown);
dom.questionInput.addEventListener('input', () => {
    autoResize();
    dom.sendBtn.disabled = !dom.questionInput.value.trim();
});

dom.newChatBtn.addEventListener('click', createNewSession);
dom.menuBtn.addEventListener('click', toggleSidebar);
dom.sidebarOverlay.addEventListener('click', closeSidebar);

dom.settingsBtn.addEventListener('click', openSettings);
dom.settingsClose.addEventListener('click', closeSettings);
dom.settingsOverlay.addEventListener('click', (e) => {
    if (e.target === dom.settingsOverlay) closeSettings();
});

dom.modelSettingsBtn.addEventListener('click', openModelSettings);
dom.modelSettingsClose.addEventListener('click', closeModelSettings);
dom.modelSettingsOverlay.addEventListener('click', (e) => {
    if (e.target === dom.modelSettingsOverlay) closeModelSettings();
});

dom.saveConfigBtn.addEventListener('click', saveConfig);

// Knowledge Base Management
console.log('[KB] Binding kbManageBtn click handler, element:', dom.kbManageBtn);
dom.kbManageBtn.addEventListener('click', () => {
    console.log('[KB] kbManageBtn clicked!');
    openKbPanel();
});
dom.kbClose.addEventListener('click', closeKbPanel);
if (dom.kbCreateBtn) dom.kbCreateBtn.addEventListener('click', createKnowledgeBase);
if (dom.kbNameInput) dom.kbNameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') createKnowledgeBase();
});
dom.kbOverlay.addEventListener('click', (e) => {
    if (e.target === dom.kbOverlay) closeKbPanel();
});
dom.kbFileInput.addEventListener('change', (e) => {
    uploadFiles(e.target.files);
    e.target.value = '';
});
dom.kbUploadArea.addEventListener('click', () => dom.kbFileInput.click());
dom.kbUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dom.kbUploadArea.classList.add('dragover');
});
dom.kbUploadArea.addEventListener('dragleave', () => {
    dom.kbUploadArea.classList.remove('dragover');
});
dom.kbUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dom.kbUploadArea.classList.remove('dragover');
    uploadFiles(e.dataTransfer.files);
});

// Escape to close panels
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeSettings();
        closeModelSettings();
        closeKbPanel();
    }
});

// ============================================================
// Init
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('[KB] DOMContentLoaded fired');
    console.log('[KB] kbManageBtn:', dom.kbManageBtn);
    console.log('[KB] kbOverlay:', dom.kbOverlay);
    console.log('[KB] kbList:', dom.kbList);
    console.log('[KB] kbSelect:', dom.kbSelect);
    dom.questionInput.focus();
    checkHealth();
    loadKnowledgeBases();
    loadSessionList();
    bindChipClicks();
    updateMenuArrow();
    initCustomSelects();
    initMiniToggles();
    setInterval(checkHealth, 30000);
});
