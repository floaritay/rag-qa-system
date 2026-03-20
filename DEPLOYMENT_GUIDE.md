# 校园课程助手部署与使用指南

## 项目概述
校园课程助手是一个基于RAG（检索增强生成）技术的智能问答系统，使用LangChain作为后端核心，Open WebUI或Gradio作为前端界面，帮助学生快速获取课程相关信息。

## 技术架构
- **后端**：LangChain + FastAPI
- **向量存储**：FAISS（本地存储）
- **模型**：百炼平台的"qwen-plus-2025-07-28"和"text-embedding-v2"
- **前端**：Open WebUI / Gradio / Web

## 部署与使用步骤

### 1. 环境准备

#### 系统要求
- Python 3.10+
- Docker desktop
- 百炼平台账号和API密钥

#### 安装依赖
```bash
# 进入项目目录
cd d:\你的文件目录

# 安装后端依赖
pip install -r backend\requirements.txt

# 安装前端依赖（Gradio）
pip install -r frontend\requirements.txt
```

### 2. 配置环境变量
```bash
# Windows
set BAILIAN_API_KEY=你的百炼平台API密钥

# Linux/Mac
export BAILIAN_API_KEY=你的百炼平台API密钥
```

### 3. 准备课程资料
将PDF格式的课程资料放入 `d:\你的文件目录\course_materials` 文件夹（如果不存在请创建）。  
支持的文件类型：PDF。

### 4. 启动后端服务
```bash
# 进入项目目录
cd d:\你的文件目录

# 启动FastAPI服务
python backend\main.py
```

服务将在 http://localhost:8001 运行

### 5. 初始化知识库（首次使用或新增课程资料时）
```bash
# 使用PowerShell
Invoke-WebRequest -Uri "http://localhost:8001/init" -Method POST

# 或使用curl
curl -X POST http://localhost:8001/init
```

如需强制重建知识库（会删除旧知识库）：
```bash
curl -X POST http://localhost:8001/init?force_rebuild=true
```
也可以在前端打开的页面点击「初始化」按钮来初始化知识库。

### 6. 部署前端界面

#### 方式一：使用 Gradio（快速验证）
```bash
# 进入frontend目录
cd d:\你的文件目录\frontend

# 启动Gradio服务
python app.py
```
访问 http://localhost:7860 即可使用

#### 方式二：启动Docker容器部署Open WebUI
启动docker desktop后，执行以下命令
```bash
# 使用powershell执行
docker run -d -p 3001:3000 -v open-webui:/app/backend/data -e OLLAMA_BASE_URL=http://host.docker.internal:11434 --name open-webui --restart always ghcr.io/open-webui/open-webui:main
# 原命令仅需执行一次（创建容器），后续用 docker start open-webui 启动即可
```
**说明**：这个命令是适配本项目的，它会：
- 使用主机网络模式，确保Open WebUI能访问本地的FastAPI服务  
- 持久化存储Open WebUI的数据  
- 配置Ollama服务地址（可选，本项目主要使用自定义API）  
- 设置容器自动重启  

打开浏览器，访问 http://localhost:3001 即可使用Open WebUI


#### 方式三：本地下载OpenWebUI
```bash
# 安装open-webui
pip install open-webui

# 启动服务
open-webui serve
```
访问 http://localhost:8080

##### 配置Open WebUI连接
1. 进入后会要求注册管理员账号
2. 点击 头像 再点击「管理员面板」
3. 进入「设置」→「外部连接」
4. 点击「添加连接」
5. 填写以下信息：
   - **连接类型**：外部
   - **认证方式**：无
   - **API类型**：OpenAI兼容API
   - **API地址**：http://localhost:8001/v1（注意：必须加 /v1）
6. 点击「保存」
7. 返回首页在模型下拉框中选择模型
8. 在聊天框中输入问题，例如：
   - "机器学习课程的成绩构成是怎样的？"
   - "数据库实验报告怎么写？"
   - "Python编程课程的重点内容有哪些？"
9. 点击发送按钮，系统会基于课程资料给出回答

### 7. 验证项目运行状态

#### 检查向量库文件

打开 `D:\你的文件目录\course_knowledge_base` 文件夹（启动服务并初始化后自动生成），能看到以下文件说明向量库创建成功：
- `index.faiss`  
- `index.pkl`

#### 测试API接口

1. **健康检查接口**：
   ```bash
   # 使用PowerShell
   Invoke-WebRequest -Uri "http://localhost:8001/health" -Method GET
   ```
   预期返回：`{"status": "healthy"}`
   ```bash
   :: 使用CMD
   curl http://127.0.0.1:8001/health
   ```
   使用浏览器访问 http://localhost:8001/health 也能看到健康状态

2. **根路径接口**：
   ```bash
   # 使用PowerShell
   Invoke-WebRequest -Uri "http://localhost:8001/" -Method GET
   ```
   预期返回：`{"message": "课程助手API已启动，访问 /ask 接口进行问答，/init 接口初始化知识库"}`
   ```bash
   :: 使用CMD
   curl http://127.0.0.1:8001/
   ```

3. **问答接口**：
   ```bash
   # 使用PowerShell
   $body = @{question="机器学习课程的成绩构成是怎样的？"} | ConvertTo-Json
   Invoke-WebRequest -Uri "http://localhost:8001/ask" -Method POST -Body $body -ContentType "application/json"
   ```
   预期返回：包含课程相关信息的JSON响应
   ```bash
   :: 使用CMD
   curl -X POST http://127.0.0.1:8001/ask -H "Content-Type: application/json" -d '{"question": "机器学习课程的成绩构成是怎样的？"}'
   :: 或
   curl -X POST -H "Content-Type: application/json" -d "{\"question\":\"多AGV调度系统研究的核心算法是什么？\"}" http://127.0.0.1:8001/ask
   ```

4. **OpenAI兼容接口**（用于Open WebUI）：
   ```bash
   # 查看可用模型
   curl http://localhost:8001/v1/models
   
   # 聊天补全（测试用）
   curl -X POST http://localhost:8001/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"course-assistant\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
   ```

## 常见问题

### 1. 初始化知识库失败  
**错误信息**：`创建向量库失败: Error code: 404 - {'error': {'code': 'InvalidEndpointOrModel.NotFound', ...}}`

**解决方案**：  
- 确保百炼平台API密钥正确
- 确保网络连接正常
- 确保 `course_materials` 文件夹中有PDF文件

### 2. API调用返回503错误   
**错误信息**：`未设置BAILIAN_API_KEY环境变量`

**解决方案**：  
- 确保已正确设置BAILIAN\_API\_KEY环境变量
- 重启FastAPI服务

### 3. Open WebUI无法连接到后端  
**解决方案**：  
- 确保FastAPI服务正在运行
- 确保API地址配置正确（http://localhost:8001/v1）
- 检查网络连接
- Open WebUI在conda环境运行，后端在全局环境运行是可以的，它们通过网络端口通信

### 4. 查询显示"在提供的课程资料中找不到相关信息"
**解决方案**：  
- 确保 `course_materials` 文件夹中有相关PDF文件
- 调用 `/init?force_rebuild=true` 强制重建知识库
- 确保问题与课程资料内容相关

## 项目结构
```
├── backend/                    # 后端服务
│   ├── main.py                # FastAPI 主服务
│   ├── knowledge_base.py      # 知识库构建模块
│   ├── qa_chain.py            # 问答链模块
│   └── requirements.txt       # Python 依赖
├── frontend/                   # 前端服务
│   ├── app.py                 # Gradio Web 界面
│   └── requirements.txt       # Python 依赖
├── course_materials/           # 课程资料存放目录（不存在需手动创建）
│   └── *.pdf                  # PDF 课程文件
├── course_knowledge_base/      # 向量库存储目录（自动生成）
│   ├── index.faiss        # FAISS 索引文件
│   └── index.pkl          # FAISS 元数据文件
├── DEPLOYMENT_GUIDE.md         # 详细部署指南
├── web/                        # 原生前端界面代码
├── solve.txt                   # 问题解决记录
└── README.md                   # 项目说明文档
```

## 技术说明
- **文档处理**：使用PyPDFLoader加载PDF文件，RecursiveCharacterTextSplitter分割文本
- **向量存储**：使用FAISS本地向量数据库，无需外部服务
- **模型调用**：使用百炼平台的qwen-plus语言模型和text-embedding-v2嵌入模型
- **API接口**：提供RESTful API和OpenAI兼容API，支持健康检查、知识库初始化和问答功能
- **CORS配置**：已配置跨域支持，允许所有来源访问

## 扩展建议
1. **添加用户认证**：为API添加身份验证，提高安全性
2. **添加多模态**：添加对Word、PPT、图片等格式的支持
3. **优化向量存储**：考虑使用更高级的向量数据库，如Milvus或Pinecone
4. **添加记忆功能**：实现模型记忆上下文功能
5. **添加多Agent架构**：
6. **部署到云服务器**：

## 故障排查

### 检查服务状态
```bash
# 检查FastAPI服务
Invoke-WebRequest -Uri "http://localhost:8001/health" -Method GET

# 检查端口占用
netstat -ano | findstr :8001
```

### 查看日志

## 联系方式

如有问题，请联系项目维护者或查看 solve.txt 文件中的问题解决记录。

---

## 附录：嵌入模型说明

嵌入模型是校园课程助手项目的核心基础，可以用一个比喻理解：

课程 PDF 里的文字是「人类语言」（比如 "机器学习成绩构成：平时 30%+ 期末 70%"），计算机看不懂；
嵌入模型的作用就是把这些「文字」转换成一串数字向量（比如 [0.123, -0.456, 0.789...]），这串数字能被计算机理解和计算；
当你提问 "机器学习成绩怎么算？" 时，系统会把问题也转换成向量，然后在向量库里找「最相似的向量」，再把对应的文字回复给你 —— 这个 "找相似" 的过程，就是 RAG 技术的核心。
简单说：嵌入模型是 "文字→数字向量" 的转换器，没有它，你的课程资料无法被检索、无法实现智能问答。

## 附录：向量库重建说明

每次启动是否会重新创建向量库？

默认情况下：不会自动重建，只有主动调用 /init 接口才会创建 / 重建向量库。

从代码来看，向量库的创建逻辑只在 init_vectorstore() 函数中，而这个函数仅在两个场景触发：
- 调用 /ask 接口时，检测到 rag_chain 未初始化 → 自动调用 init_vectorstore()（首次问答时）；
- 主动调用 /init 接口 → 强制调用 init_vectorstore()。

单纯执行 python .\backend\main.py 启动服务时，只会：
- 启动 FastAPI 服务（监听 8001 端口）；
- 加载全局变量（如 bailian_api_key）；
- 不会执行任何向量库创建逻辑，也就不会调用嵌入模型。

## 附录：课程资料更新说明

course_materials 文件夹文件变动后，会自动更新向量库吗？

不会自动更新！代码中没有「监听文件变动→自动重建向量库」的逻辑，分情况处理：

新增 / 修改 / 删除文件后，想更新向量库
需主动调用 /init 接口，触发 init_vectorstore() 重新加载文件→分割→生成向量→覆盖旧向量库。
