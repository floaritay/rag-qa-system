# 模型配置指南

本项目支持两种模型提供商：**硅基流动（SiliconFlow）** 和 **自定义 OpenAI 兼容 API**。三个模型服务（LLM、Embedding、Reranker）可以独立选择不同的提供商，实现混合使用。

---

## 快速开始：使用硅基流动（默认）

硅基流动提供免费模型，只需一个 API Key 即可使用全部三个模型服务。

### 1. 获取 API Key

前往 [siliconflow.cn](https://cloud.siliconflow.cn/account/ak) 注册并获取 API Key。

### 2. 配置

1. 打开前端页面，点击左侧边栏的 **模型设置**
2. 在顶部 **硅基流动 API Key** 输入框中填入你的 Key
3. 系统会自动从硅基流动 API 获取可用模型列表（包含免费和付费模型），填充到下方的下拉框中
4. 分别为 LLM、Embedding、Reranker 选择模型
5. 点击 **保存配置**

### 默认模型

| 服务 | 默认模型 | 说明 |
|------|---------|------|
| LLM | Qwen/Qwen3-8B | 通义千问 8B，免费 |
| Embedding | BAAI/bge-m3 | 向量嵌入模型，免费 |
| Reranker | BAAI/bge-reranker-v2-m3 | 重排序模型，免费 |

### .env 配置示例

```env
SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=Qwen/Qwen3-8B
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_BASE_URL=https://api.siliconflow.cn/v1
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

硅基流动模式下，三个模型共用 `SILICONFLOW_API_KEY`，Base URL 固定为 `https://api.siliconflow.cn/v1`。

---

## 使用自定义 OpenAI 兼容 API

如果你有自己的 LLM 服务（如 OpenAI、阿里云百炼、本地 Ollama 等），可以为任意模型切换到自定义模式。

### 配置步骤

1. 打开前端页面，点击左侧边栏的 **模型设置**
2. 找到要自定义的模型（LLM / Embedding / Reranker）
3. 点击右侧的 **自定义** 按钮切换到自定义模式
4. 填入三项信息：
   - **模型名称**：API 提供商的模型 ID，如 `gpt-4o`、`qwen-plus`、`deepseek-chat`
   - **Base URL**：API 地址，如 `https://api.openai.com/v1`、`https://dashscope.aliyuncs.com/compatible-mode/v1`
   - **API Key**：对应服务的密钥
5. 点击 **保存配置**

### 常见 API 地址

| 提供商 | Base URL |
|--------|----------|
| OpenAI | `https://api.openai.com/v1` |
| 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 硅基流动 | `https://api.siliconflow.cn/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |
| Ollama（本地） | `http://localhost:11434/v1` |

### .env 配置示例

自定义模式下，每个模型使用各自的 API Key 和 Base URL：

```env
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx

EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_API_KEY=sk-xxxxxxxxxxxxxxxx

RERANKER_BASE_URL=https://api.siliconflow.cn/v1
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_API_KEY=sk-xxxxxxxxxxxxxxxx
```

---

## 混合使用

三个模型可以自由组合不同的提供商。例如：

- **LLM** 使用阿里云百炼的 `qwen-plus`（自定义模式）
- **Embedding** 使用硅基流动的 `BAAI/bge-m3`（硅基流动模式）
- **Reranker** 使用硅基流动的 `BAAI/bge-reranker-v2-m3`（硅基流动模式）

前端会根据每个模型的 Base URL 自动判断其提供商类型：
- Base URL 包含 `siliconflow` → 显示为硅基流动模式
- 其他 URL → 显示为自定义模式

重启后端后，前端会自动读取配置并正确显示每个模型的提供商状态。

---

## API 端点

### 获取配置

```
GET /config
```

返回当前所有模型配置，API Key 以掩码显示（如 `sk-***abcd`）。

### 更新配置

```
POST /config
```

请求体格式：

```json
{
  "siliconflow_api_key": "sk-xxx",
  "models": {
    "llm": {
      "model": "Qwen/Qwen3-8B",
      "base_url": "https://api.siliconflow.cn/v1",
      "api_key": ""
    },
    "embedding": {
      "model": "BAAI/bge-m3",
      "base_url": "https://api.siliconflow.cn/v1",
      "api_key": ""
    },
    "reranker": {
      "model": "BAAI/bge-reranker-v2-m3",
      "base_url": "https://api.siliconflow.cn/v1",
      "api_key": ""
    }
  }
}
```

- `siliconflow_api_key`：硅基流动共享 Key，更新后所有使用硅基流动的模型自动使用此 Key
- 每个模型的 `api_key`：留空或以 `***` 开头表示不修改；非空且非掩码则更新为新值
- Base URL 为空时保留现有值
- 配置会自动持久化到 `.env` 文件

### 获取硅基流动可用模型

```
GET /models/siliconflow
Header: X-API-Key: sk-xxx
```

从硅基流动 API 获取可用模型列表，按类型分类（llm / embedding / reranker）。列表包含免费和付费模型，请在 [siliconflow.cn/pricing](https://siliconflow.cn/pricing) 确认具体模型的定价。如果 API 调用失败，返回内置的免费模型兜底列表。

---

## 配置持久化与启动加载

配置保存后会写入项目根目录的 `.env` 文件。后端启动时按以下逻辑加载：

1. 读取 `SILICONFLOW_API_KEY` 作为共享 Key
2. 读取每个模型的 `BASE_URL`，如果 `.env` 中没有则默认为硅基流动地址
3. 根据 Base URL 决定 API Key 来源：
   - URL 包含 `siliconflow` → 使用 `SILICONFLOW_API_KEY`
   - 其他 URL → 使用对应的 `LLM_API_KEY` / `EMBEDDING_API_KEY` / `RERANKER_API_KEY`

修改 `.env` 后需要重启后端才能生效。通过前端保存配置会立即生效并自动写入 `.env`。

---

## 注意事项

- 硅基流动 API 返回的模型列表包含免费和付费模型，系统已过滤 `Pro/` 和 `LoRA/` 前缀的付费模型，但其他付费模型仍可能出现在列表中。请在 [siliconflow.cn/pricing](https://siliconflow.cn/pricing) 确认定价。API 调用失败时使用内置的免费模型兜底列表
- 切换 Embedding 模型后，需要重建知识库（`POST /init?force_rebuild=true`），因为不同模型的向量维度不同
- 切换 LLM 或 Reranker 模型不需要重建知识库，立即生效
- 自定义模式下 API Key 留空表示"不修改"，会保留上次保存的值
