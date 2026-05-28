# 多知识库支持（Multi-Knowledge-Base）— 实现原理与使用说明

## 概述

系统支持多知识库管理，用户可以创建、切换、重命名和删除独立的知识库。每个知识库拥有独立的文档集合和向量索引，互不干扰。会话（Session）与知识库绑定，切换知识库时自动过滤对应的会话列表。

## 目录结构

```
knowledge_bases/
  registry.json                  # 知识库注册表（主索引）
  default/                       # 默认知识库（从旧数据自动迁移）
    index.faiss                  # FAISS 向量索引
    index.pkl                    # FAISS 索引元数据
    bm25_index.pkl               # BM25 关键词索引
    manifest.json                # 文件索引记录（mtime + chunks）
    materials/                   # 源文档（PDF/PPTX/DOCX/MD）
  <kb-slug>/                     # 用户创建的知识库
    index.faiss
    index.pkl
    bm25_index.pkl
    manifest.json
    materials/
```

### registry.json

所有知识库的元数据注册表：

```json
{
  "default": {
    "id": "default",
    "name": "默认知识库",
    "created_at": "2026-05-26T10:00:00",
    "updated_at": "2026-05-26T10:00:00",
    "description": ""
  },
  "machine-learning": {
    "id": "machine-learning",
    "name": "机器学习",
    "created_at": "2026-05-26T12:00:00",
    "updated_at": "2026-05-26T12:00:00",
    "description": "ML 相关课程资料"
  }
}
```

## 实现原理

### KBManager 类

核心管理类，负责知识库的生命周期管理和内存缓存：

```
KBManager
├── 注册表管理
│   ├── load_registry()      # 读取 registry.json
│   ├── save_registry()      # 写入 registry.json
│   ├── list_kbs()           # 列出所有知识库（含文件数）
│   ├── create_kb()          # 创建知识库目录 + 注册
│   ├── delete_kb()          # 删除目录 + 注销（禁止删除 default）
│   └── rename_kb()          # 更新注册表中的名称
│
├── 路径解析
│   ├── get_kb_path(kb_id)        # knowledge_bases/<kb_id>/
│   └── get_materials_path(kb_id) # knowledge_bases/<kb_id>/materials/
│
└── 内存缓存（LRU）
    ├── get(kb_id)           # 获取知识库数据（懒加载 + LRU 淘汰）
    ├── invalidate(kb_id)    # 使缓存失效（索引变更后调用）
    └── _load_from_disk()    # 从磁盘加载 FAISS + BM25
```

**LRU 缓存策略**：最多同时加载 2 个知识库到内存（`max_loaded=2`）。访问时更新优先级，超出上限时淘汰最久未访问的知识库。使用 `threading.Lock` 保证线程安全。

**缓存数据结构**：

```python
{
    "vectorstore": FAISS,           # 向量存储
    "bm25_index": BM25Okapi,        # BM25 索引（可为 None）
    "bm25_docs": list[Document],     # BM25 文档列表（可为 None）
    "retriever": VectorStoreRetriever # 检索器
}
```

### 初始化

启动时 `init_default_kb()` 自动确保默认知识库目录结构和 `registry.json` 存在。旧版 `course_knowledge_base/` 和 `course_materials/` 目录已迁移并删除，所有数据统一存放在 `knowledge_bases/` 下。

### 会话关联

数据库 `sessions` 表新增 `kb_id` 列：

```sql
ALTER TABLE sessions ADD COLUMN kb_id TEXT DEFAULT 'default';
```

- 新建会话时记录当前 `kb_id`
- 查询会话时可按 `kb_id` 过滤
- 切换知识库时只显示对应会话

### 检索流程

所有检索函数（`hybrid_retrieve`、`rag_query`、`rag_query_stream` 等）新增 `kb_id` 参数：

```
用户提问
    │
    ▼
解析 kb_id（请求体 > 会话关联 > 默认 "default"）
    │
    ▼
kb_manager.get(kb_id)
    ├── 缓存命中 → 直接返回
    └── 缓存未命中 → 从磁盘加载 → LRU 缓存
    │
    ▼
使用该知识库的 vectorstore + bm25_index 执行检索
    │
    ▼
返回结果
```

## API 端点

### 知识库管理

#### 列出所有知识库

```bash
GET /knowledge-bases
```

响应：
```json
[
  {
    "id": "default",
    "name": "默认知识库",
    "created_at": "2026-05-26T10:00:00",
    "updated_at": "2026-05-26T10:00:00",
    "description": "",
    "file_count": 5
  },
  {
    "id": "machine-learning",
    "name": "机器学习",
    "created_at": "2026-05-26T12:00:00",
    "updated_at": "2026-05-26T12:00:00",
    "description": "ML 相关课程资料",
    "file_count": 3
  }
]
```

#### 创建知识库

```bash
POST /knowledge-bases
Content-Type: application/json

{
  "name": "机器学习",
  "description": "ML 相关课程资料"
}
```

响应：
```json
{
  "id": "machine-learning",
  "name": "机器学习",
  "created_at": "2026-05-26T12:00:00",
  "updated_at": "2026-05-26T12:00:00",
  "description": "ML 相关课程资料",
  "file_count": 0
}
```

`id` 由 `name` 自动生成（slug 化：中文保留，特殊字符转 `-`，小写化）。重名时追加时间戳后缀。

#### 获取知识库详情

```bash
GET /knowledge-bases/{kb_id}
```

#### 重命名知识库

```bash
PUT /knowledge-bases/{kb_id}
Content-Type: application/json

{"name": "新名称"}
```

#### 删除知识库

```bash
DELETE /knowledge-bases/{kb_id}
```

- 不允许删除 `default` 知识库
- 删除会清理磁盘上的整个目录
- 关联的会话不会被删除，但无法再检索

#### 强制重建知识库

```bash
POST /knowledge-bases/{kb_id}/rebuild
```

删除现有索引，从 `materials/` 目录重新构建向量库和 BM25 索引。

### 已有端点的 kb_id 参数

以下端点新增 `kb_id` 查询参数（默认 `"default"`）：

| 端点 | 参数位置 | 说明 |
|------|---------|------|
| `POST /init` | query: `?kb_id=xxx` | 初始化指定知识库的向量库 |
| `GET /materials` | query: `?kb_id=xxx` | 列出指定知识库的文件 |
| `POST /materials/upload` | query: `?kb_id=xxx` | 上传文件到指定知识库 |
| `POST /materials/index` | query: `?kb_id=xxx` | 对指定知识库执行增量索引 |
| `DELETE /materials/{filename}` | query: `?kb_id=xxx` | 从指定知识库删除文件 |

以下端点在请求体中新增 `kb_id` 字段：

| 端点 | 说明 |
|------|------|
| `POST /ask` | `kb_id` 为可选，未提供时从会话中解析 |
| `POST /ask/stream` | 同上 |
| `POST /v1/chat/completions` | 同上 |
| `POST /sessions` | 创建会话时绑定 `kb_id`（默认 `"default"`） |
| `GET /sessions` | query: `?kb_id=xxx` 过滤会话列表 |

## 前端使用

### 知识库选择器

侧边栏顶部显示当前知识库名称，点击可展开下拉菜单切换知识库：

```
┌─────────────────────┐
│ 📂 默认知识库    ▾ │  ← 点击切换
├─────────────────────┤
│ 📂 默认知识库       │
│ 📂 机器学习         │
│ 📂 自然语言处理     │
└─────────────────────┘
```

切换知识库时：
- 清空当前聊天区域
- 显示欢迎页面
- 重新加载该知识库的会话列表

### 知识库管理面板

点击侧边栏底部的设置图标，选择"知识库管理"进入管理面板。面板分为两个区域：

**上半部分 — 知识库列表**：
- 显示所有知识库及其文件数
- 当前选中的知识库高亮
- 每个知识库有重命名和删除按钮
- 底部有创建新知识库的输入框

**下半部分 — 文件管理**：
- 标题显示当前选中的知识库名称
- 拖拽上传区域
- 文件列表（上传、删除、重建索引）

操作流程：
1. 在知识库列表中选择目标知识库
2. 在文件管理区域上传/删除文档
3. 点击"重建索引"使变更生效

## 使用场景

| 场景 | 操作 |
|------|------|
| 按课程分类资料 | 为每门课程创建独立知识库 |
| 项目文档隔离 | 不同项目使用不同知识库 |
| 测试新文档 | 创建临时知识库测试，不影响主库 |
| 团队共享 | 每人维护自己的知识库 |

## 注意事项

- **default 知识库不可删除**：系统保证始终存在一个默认知识库
- **LRU 缓存上限**：最多同时加载 2 个知识库到内存，频繁切换时会有短暂的加载延迟
- **会话关联**：会话创建时绑定知识库，之后不会自动变更。删除知识库不会删除关联会话
- **索引独立**：每个知识库有独立的 FAISS 和 BM25 索引，互不影响
- **目录结构**：所有知识库数据统一存放在 `knowledge_bases/` 下，旧版目录已迁移删除
