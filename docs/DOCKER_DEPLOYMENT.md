# Docker 部署指南

## 快速开始

### 前置条件

- 已安装 [Docker](https://docs.docker.com/get-docker/)
- 一个硅基流动 API Key（[获取地址](https://cloud.siliconflow.cn/)）

### 方式一：docker run（最简）

```bash
# 创建工作目录和配置
mkdir rag && cd rag
echo "SILICONFLOW_API_KEY=你的API密钥" > .env

# 启动服务
docker run -d --name rag -p 80:80 \
  --env-file .env \
  -v ./knowledge_bases:/app/knowledge_bases \
  ghcr.io/floaritay/rag-qa-system:main

# 初始化知识库
curl -X POST http://localhost/api/init
```

### 方式二：Docker Compose

```bash
mkdir rag && cd rag

# 下载配置文件
curl -O https://raw.githubusercontent.com/floaritay/rag-qa-system/main/docker-compose.yml

# 配置 API Key
echo "SILICONFLOW_API_KEY=你的API密钥" > .env

# 启动服务
docker-compose up -d

# 初始化知识库
curl -X POST http://localhost/api/init
```

启动后访问 http://localhost 即可使用。

### 知识库初始化

启动后需要初始化向量索引：

```bash
curl -X POST http://localhost/api/init
```

后续添加或修改了文档，需要重建索引：

```bash
curl -X POST "http://localhost/api/init?force_rebuild=true"
```

## 常用命令

```bash
# 查看日志
docker logs -f rag            # docker run 方式
docker-compose logs -f        # docker-compose 方式

# 停止服务
docker rm -f rag              # docker run 方式
docker-compose down           # docker-compose 方式

# 更新镜像
docker pull ghcr.io/floaritay/rag-qa-system:main
# 然后重新执行 docker run 或 docker-compose up -d

# 查看运行状态
docker ps
```

## 环境变量说明

在 `.env` 文件中配置，或通过 `docker run -e` 传入：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `SILICONFLOW_API_KEY` | 硅基流动 API Key（三个服务共用） | 必填 |
| `LLM_BASE_URL` | LLM API 地址 | `https://api.siliconflow.cn/v1` |
| `LLM_MODEL` | LLM 模型名称 | `Qwen/Qwen3-8B` |
| `EMBEDDING_BASE_URL` | Embedding API 地址 | `https://api.siliconflow.cn/v1` |
| `EMBEDDING_MODEL` | Embedding 模型名称 | `BAAI/bge-m3` |
| `RERANKER_BASE_URL` | Reranker API 地址 | `https://api.siliconflow.cn/v1` |
| `RERANKER_MODEL` | Reranker 模型名称 | `BAAI/bge-reranker-v2-m3` |

**API Key 优先级**：各服务独立 KEY (`LLM_API_KEY` / `EMBEDDING_API_KEY` / `RERANKER_API_KEY`) > `SILICONFLOW_API_KEY`。当所有服务使用硅基流动时，只需设置 `SILICONFLOW_API_KEY`。

## 架构说明

```
Docker 容器 (端口 80)
├── Nginx (:80)
│   ├── /         → 静态前端 (HTML/CSS/JS)
│   └── /api/     → 反向代理 → FastAPI (:8001)
├── FastAPI 后端 (:8001) — 由 supervisord 管理
└── 数据卷挂载
    ├── .env                → 环境变量配置
    └── knowledge_bases/    → 向量索引与文档
```

- **Nginx**：统一入口，处理静态文件和 API 反向代理，支持 SSE 流式响应
- **supervisord**：进程管理器，同时运行 nginx 和 FastAPI 后端
- **数据持久化**：`knowledge_bases/` 目录通过 volume 挂载到宿主机，容器重建不丢失

## 本地构建（开发调试）

如需修改代码后本地构建镜像：

```bash
git clone https://github.com/floaritay/rag-qa-system.git
cd rag-qa-system

# 编辑 docker-compose.yml，取消 build 行的注释
docker-compose up -d --build
```

## 自定义端口

默认映射到宿主机 80 端口。如需修改：

```bash
# docker run 方式：改 -p 参数
docker run -d --name rag -p 8080:80 ...

# docker-compose 方式：编辑 docker-compose.yml
# ports:
#   - "8080:80"
```

## 数据备份

重要数据位于以下位置：

- `knowledge_bases/` — 向量索引与源文档（已通过 volume 挂载到宿主机）
- `.env` — API 配置（已通过 volume 挂载到宿主机）

会话历史存储在容器内部，重建容器会丢失（会话本身 7 天自动过期）。

## 故障排查

**容器启动失败**
```bash
docker logs rag
```

**API 无响应**
```bash
# 检查后端是否正常运行
docker exec rag curl -s http://127.0.0.1:8001/health
```

**重建容器后数据丢失**
确认 `knowledge_bases/` 目录存在于宿主机工作目录下。该目录通过 volume 挂载，不会因容器重建而丢失。
