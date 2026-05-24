# 增量索引（Incremental Index）— 实现原理与使用说明

## 概述

系统支持增量索引，新增文档时只对新文件做 embedding 并合并到现有索引，删除文档时只移除对应文件的向量，均无需全量重建。同时保留全量重建能力，三者可按需选择。

## 实现原理

### 新增文档流程

```
上传新文件 / 手动触发
        │
        ▼
┌───────────────────────────────────────────────┐
│  扫描 course_materials/                        │
│  对比 manifest.json 中已记录的文件 mtime        │
│  筛选出新增或修改的文件                         │
└──────────────────┬────────────────────────────┘
                   │ 仅新文件
                   ▼
┌───────────────────────────────────────────────┐
│  加载 → 分块 → embedding → 临时 FAISS 索引     │
└──────────────────┬────────────────────────────┘
                   │ merge_from
                   ▼
┌───────────────────────────────────────────────┐
│  合并到现有 FAISS 索引 + 追加 BM25 索引        │
│  保存 index.faiss / index.pkl / bm25_index.pkl │
│  更新 manifest.json                           │
└───────────────────────────────────────────────┘
```

### 删除文档流程

```
删除文件
    │
    ▼
┌───────────────────────────────────────────────┐
│  遍历 FAISS docstore                           │
│  按 metadata["source"] 匹配文件名              │
│  收集该文件的所有 docstore ID                   │
└──────────────────┬────────────────────────────┘
                   │ ids_to_delete
                   ▼
┌───────────────────────────────────────────────┐
│  vectorstore.delete(ids)                       │
│  → 从 FAISS 索引移除向量                        │
│  → 从 docstore 移除文档                         │
│  → 重建 index_to_docstore_id 映射              │
└──────────────────┬────────────────────────────┘
                   ▼
┌───────────────────────────────────────────────┐
│  重建 BM25 索引（排除已删除文件的文档）          │
│  更新 manifest.json（移除该文件条目）            │
└───────────────────────────────────────────────┘
```

### manifest.json

增量索引的核心是 `course_knowledge_base/manifest.json`，记录每个已索引文件的修改时间和分块数：

```json
{
  "files": {
    "机器学习基础.pdf": {"mtime": 1716500000.0, "chunks": 42},
    "lecture_notes.md": {"mtime": 1716400000.0, "chunks": 8}
  }
}
```

每次增量索引时，对比磁盘文件的 `mtime` 与 manifest 中记录的值：
- **新文件**：manifest 中不存在 → 需要索引
- **修改的文件**：mtime 不同 → 需要重新索引
- **未变化的文件**：mtime 一致 → 跳过

### 兼容性处理

首次使用增量索引时，如果已有旧的向量库但没有 manifest（全量重建生成的旧数据），系统会自动执行一次全量重建并生成 manifest，后续即可正常使用增量索引。

## API 端点

### 自动索引：POST /materials/upload

上传文件后自动触发增量索引，无需手动操作。

```bash
curl -X POST http://localhost:8001/materials/upload \
  -F "files=@新文档.pdf"
```

响应示例：
```json
{
  "status": "success",
  "uploaded": ["新文档.pdf"],
  "index": {
    "status": "success",
    "indexed": [{"file": "新文档.pdf", "chunks": 15}],
    "message": "增量索引完成，处理 1 个文件"
  }
}
```

### 手动索引：POST /materials/index

手动触发增量索引，适合批量上传后统一索引的场景。

```bash
curl -X POST http://localhost:8001/materials/index
```

响应示例：
```json
{
  "status": "success",
  "indexed": [
    {"file": "新文档1.pdf", "chunks": 15},
    {"file": "新文档2.docx", "chunks": 8}
  ],
  "message": "增量索引完成，处理 2 个文件"
}
```

无新文件时：
```json
{
  "status": "success",
  "indexed": [],
  "message": "无新增或修改的文件"
}
```

### 删除文件：DELETE /materials/{filename}

删除文件时自动从索引中移除对应向量，无需全量重建。

```bash
curl -X DELETE http://localhost:8001/materials/旧文档.pdf
```

响应示例：
```json
{
  "status": "success",
  "message": "已删除: 旧文档.pdf",
  "index": {
    "status": "success",
    "removed": 12,
    "message": "已移除 旧文档.pdf 的 12 个向量"
  }
}
```

索引中未找到该文件的向量时（例如从未索引过）：
```json
{
  "status": "success",
  "message": "已删除: 旧文档.pdf",
  "index": {
    "status": "success",
    "removed": 0,
    "message": "索引中未找到 旧文档.pdf 的向量"
  }
}
```

### 全量重建：POST /init?force_rebuild=true

删除整个向量库，从所有文件重新构建。重建后自动生成新的 manifest。

```bash
curl -X POST "http://localhost:8001/init?force_rebuild=true"
```

## 使用场景对比

| 场景 | 推荐方式 | 说明 |
|------|---------|------|
| 上传 1-2 个新文件 | 自动索引（上传即生效） | 上传接口自动触发增量索引 |
| 批量上传多个文件 | 手动索引 | 全部上传后调用一次 `/materials/index` |
| 删除文件 | 自动清理（删除即生效） | 删除接口自动从索引中移除对应向量 |
| 替换已有文件 | 自动索引 | mtime 变化会触发重新索引该文件 |
| 首次使用 | 全量重建 | `/init` 初始化 |

## 注意事项

- **同名文件覆盖**：上传同名文件会覆盖原文件，增量索引检测到 mtime 变化后会重新索引该文件
- **manifest 损坏**：如果 manifest.json 被手动删除或损坏，下次增量索引会自动触发全量重建
- **删除后 BM25 重建**：删除文件时需要重建 BM25 索引（排除已删除文件的文档），文档量大时可能有短暂延迟
