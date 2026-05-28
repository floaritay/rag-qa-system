# Demo 演示指南

## 概述

Demo 是一个纯静态页面版本的个人知识库，无需后端服务即可展示完整的 RAG 问答交互流程。所有 API 调用通过前端 Mock 拦截器返回模拟数据，适合用于产品演示、GitHub Pages 部署或离线展示。

**在线地址**：https://floaritay.github.io/rag-qa-system/

## 文件结构

```
demo/
├── index.html      # 入口页面
├── styles.css      # 样式文件（与 web/ 一致）
├── app.js          # 应用逻辑 + Mock Fetch 拦截器
└── mock-data.js    # 模拟数据（知识库、会话、问答等）
```

## 快速启动

### 方式一：Python HTTP 服务器

```bash
python -m http.server 8080 --directory demo
```

浏览器访问 `http://localhost:8080`。

### 方式二：直接打开

双击 `demo/index.html` 可直接在浏览器中打开（部分浏览器可能限制本地文件的 JS 加载）。

## 演示内容

### 预设问答

欢迎页提供三个快捷问题，点击即可体验流式输出效果：

| 问题 | 展示内容 |
|------|----------|
| 课程主要内容 | RAG 技术四大模块概览，展示多级标题和列表渲染 |
| 知识点总结 | 混合检索、查询优化、重排序等近期知识点 |
| 重要概念 | 表格形式展示核心概念清单，含重要度标记 |

### 自由提问

在输入框中输入任意问题，系统会返回通用的模拟回答。输入内容如果与预设问题部分匹配，会返回对应的预设回答。

### 知识库切换

侧边栏的知识库选择器支持在「默认知识库」和「技术文档」之间切换，切换后会话列表会自动更新。

### 会话管理

- **新建对话**：点击侧边栏「新建对话」按钮
- **切换会话**：点击侧边栏中的会话项，加载对应的历史消息
- **删除会话**：点击会话项右侧的删除按钮

### 检索策略设置

点击侧边栏底部的「检索策略」按钮，可切换检索方式、预检索优化和后检索优化。策略冲突（HyDE + 混合检索）会显示警告提示。

### 模型设置

点击侧边栏底部的「模型设置」按钮，可查看模型配置面板。支持硅基流动和自定义两种模式切换，保存操作会显示 "Demo 模式" 提示。

## Mock 拦截原理

`app.js` 顶部的 Mock Fetch 拦截器通过替换全局 `window.fetch` 实现：

```
用户操作 → fetch(url) → Mock 拦截器 → 匹配 URL 路径 → 返回模拟 Response
```

### 拦截的端点

| 端点 | 方法 | 行为 |
|------|------|------|
| `/health` | GET | 返回 `{ status: "ok" }` |
| `/ask/stream` | POST | 返回 SSE 流（sources + 逐字 token） |
| `/sessions` | GET/POST | 内存中的会话 CRUD |
| `/sessions/{id}` | GET/DELETE | 会话详情与删除 |
| `/sessions/{id}/messages` | GET | 返回会话消息列表 |
| `/knowledge-bases` | GET/POST | 知识库列表与创建 |
| `/materials` | GET | 文件列表 |
| `/materials/upload` | POST | 显示 "不支持上传" 提示 |
| `/config` | GET/POST | 模型配置读写 |
| `/models/siliconflow` | GET | 硅基流动模型列表 |

### SSE 流式模拟

`/ask/stream` 端点通过 `ReadableStream` 实现流式输出：

1. 先发送 `sources` 事件（检索来源文档）
2. 将回答文本按字符拆分为 token（中文逐字，英文每 3 字符）
3. 每个 token 间隔 35-65ms 发送，模拟真实打字效果

## 部署到 GitHub Pages

本项目使用 GitHub Actions 自动部署。每次 `main` 分支的 `demo/` 目录发生变更时，会自动部署到 GitHub Pages。

**在线地址**：https://floaritay.github.io/rag-qa-system/

### 部署配置

Workflow 文件位于 `.github/workflows/deploy-demo.yml`，已包含在仓库中。

首次部署需要在 GitHub 仓库设置中启用：

1. 打开仓库 Settings → Pages
2. **Source** 选择 **GitHub Actions**
3. 保存

### 日常使用

修改 `demo/` 目录后，只需提交推送到 `main`，GitHub Actions 会自动重新部署：

```bash
git add demo/
git commit -m "update: 更新 Demo"
git push github main
```

部署状态可在仓库 **Actions** 标签页查看，通常 1-2 分钟完成。

## 常见问题

**Q: 部署后页面空白？**
A: 打开浏览器开发者工具（F12）查看 Console 是否有 JS 报错。常见原因是路径问题——确保 `index.html` 中引用的 `mock-data.js`、`app.js`、`styles.css` 使用相对路径。

**Q: 样式或字体没有加载？**
A: Demo 使用了 Google Fonts（Outfit、DM Sans），需要网络连接。如果在内网环境，可以将字体下载到本地并修改 `styles.css` 中的 `@import` 路径。

**Q: 如何自定义 Demo 内容？**
A: 编辑 `demo/mock-data.js` 文件：
- 修改 `knowledgeBases` 数组添加/删除知识库
- 修改 `presetQA` 对象添加新的预设问答
- 修改 `sessions` 和 `messages` 调整会话历史
- 修改 `defaultAnswer` 自定义通用回答

**Q: 如何更新已部署的 Demo？**
A: 修改 `demo/` 目录后提交推送到 `main`，GitHub Actions 会自动重新部署（见上方"日常使用"）。

## 与正式版的区别

| 特性 | 正式版 | Demo |
|------|--------|------|
| 后端服务 | 需要 FastAPI 后端 | 纯静态，无需后端 |
| 数据持久化 | SQLite + FAISS | 内存，刷新即重置 |
| 文档检索 | 真实向量检索 | 模拟数据 |
| LLM 回答 | 真实 API 调用 | 预写回答 |
| 文件上传 | 支持 | 提示不可用 |
