const API_URL = 'http://127.0.0.1:8001';

// ============================================================
// State
// ============================================================
let isProcessing = false;
let currentSessionId = null;

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
    rebuildBtn: $('rebuildBtn'),
    retrievalStrategySelect: $('retrievalStrategySelect'),
    preRetrievalSelect: $('preRetrievalSelect'),
    postRetrievalSelect: $('postRetrievalSelect'),
    strategyConflict: $('strategyConflict'),
    strategyConflictMsg: $('strategyConflictMsg'),
    toastContainer: $('toastContainer'),
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
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    dom.toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 250);
    }, 3000);
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
        const res = await fetch(`${API_URL}/sessions`);
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
            body: JSON.stringify({ title: '新对话' }),
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
            dom.topbarTitle.textContent = '智能课程助手';
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
function showWelcome() {
    dom.chatMessages.innerHTML = `
        <div class="welcome" id="welcome">
            <div class="welcome-glow"></div>
            <div class="welcome-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                    <path d="M2 17l10 5 10-5"/>
                    <path d="M2 12l10 5 10-5"/>
                </svg>
            </div>
            <h2>智能课程助手</h2>
            <p>基于 RAG 技术，精准检索课程资料，为您解答专业问题</p>
            <div class="welcome-chips">
                <button class="chip" data-question="这门课程的主要内容是什么？">课程主要内容</button>
                <button class="chip" data-question="请总结一下最近讲的知识点">知识点总结</button>
                <button class="chip" data-question="有哪些重要的概念需要掌握？">重要概念</button>
            </div>
        </div>
    `;
    bindChipClicks();
}

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
                    body: JSON.stringify({ title: '新对话' }),
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

        const body = { question };
        if (currentSessionId) body.session_id = currentSessionId;
        body.retrieval_strategy = getSelectValue('retrievalStrategySelect');
        body.pre_retrieval = getSelectValue('preRetrievalSelect');
        body.post_retrieval = getSelectValue('postRetrievalSelect');

        const res = await fetch(`${API_URL}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        removeTypingIndicator();

        if (res.ok) {
            const result = await res.json();
            const answer = result.answer || '抱歉，无法获取回答';
            const sources = result.sources || [];
            addMessage(answer, 'assistant', true, sources);

            if (result.session_id && !currentSessionId) {
                currentSessionId = result.session_id;
            }
            loadSessionList();
        } else {
            const err = await res.json().catch(() => ({}));
            addErrorMessage(`请求失败：${err.detail || '未知错误'}`);
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
async function rebuildKnowledgeBase() {
    dom.rebuildBtn.disabled = true;
    const label = dom.rebuildBtn.querySelector('span');
    const originalText = label.textContent;
    label.textContent = '重建中...';

    try {
        const res = await fetch(`${API_URL}/init?force_rebuild=true`, { method: 'POST' });
        if (res.ok) {
            const result = await res.json();
            showToast(result.message || '知识库重建成功', 'success');
        } else {
            const err = await res.json().catch(() => ({}));
            showToast(`重建失败：${err.detail || '未知错误'}`, 'error');
        }
    } catch {
        showToast('无法连接到后端服务', 'error');
    }

    label.textContent = originalText;
    dom.rebuildBtn.disabled = false;
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
    // Close all open selects
    document.querySelectorAll('.custom-select.open').forEach(s => s.classList.remove('open'));
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

dom.rebuildBtn.addEventListener('click', rebuildKnowledgeBase);

// Escape to close settings
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeSettings();
});

// ============================================================
// Init
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    dom.questionInput.focus();
    checkHealth();
    loadSessionList();
    bindChipClicks();
    updateMenuArrow();
    initCustomSelects();
    setInterval(checkHealth, 30000);
});
