import os
import json
import datetime
import asyncio
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import StreamingResponse, HTMLResponse
import chromadb
import PyPDF2
import uvicorn
from dotenv import load_dotenv

# 导入 SQLAlchemy 与 Redis
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import redis

# 导入 LangChain 智能体组件
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents.agent import AgentExecutor
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
load_dotenv()

app = FastAPI()
api_key = os.getenv("DEEPSEEK_API_KEY")

# 1. 数据库配置
raw_db_url = os.getenv("DATABASE_URL", "sqlite:///./chat_history.db")
database_url = raw_db_url.replace("postgres://", "postgresql://", 1) if raw_db_url.startswith("postgres://") else raw_db_url
connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
engine = create_engine(database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class History(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 2. Redis 配置
redis_url = os.getenv("REDIS_URL")
redis_client = redis.Redis.from_url(redis_url, decode_responses=True) if redis_url else None
if redis_client:
    print("✅ 成功连接到云端 Redis 内存加速引擎！")

# 3. 向量数据库
chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_or_create_collection(name="zhexin_kb")


@tool
def search_knowledge_base(query: str) -> str:
    """当用户提问关于上传的私有文档内容时，必须调用此工具进行检索。"""
    print(f"\n[🛠️ 工具调用] 正在检索本地知识库: {query}")
    results = collection.query(query_texts=[query], n_results=2)
    if results['documents'] and results['documents'][0]:
        return "\n\n".join(results['documents'][0])
    return "知识库中未找到相关内容。"

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    print(f"\n[🛠️ 工具调用] 正在查询 {city} 的天气...")
    return f"{city}的天气是多云转晴，25℃，空气质量优，适合出行。"

@tool
def get_current_time() -> str:
    """获取当前系统时间"""
    time_str = datetime.datetime.now().strftime("%Y年%m月%d日 %H时%M分")
    print(f"\n[🛠️ 工具调用] 获取系统时间: {time_str}")
    return f"现在的系统时间是：{time_str}。"

tools = [search_knowledge_base, get_weather, get_current_time]

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个万能的全栈AI助理。你可以自主决定是否使用工具查询天气、时间或检索私有知识库。请用Markdown格式输出，保持专业友善。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <title>Agentic RAG 智能企业知识库 | 李哲鑫</title>
        <script src="https://cdn.bootcdn.net/ajax/libs/marked/4.2.12/marked.min.js"></script>
        <style>
            :root { --primary-color: #10a37f; --bg-color: #f3f4f6; --chat-bg: #ffffff; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: var(--bg-color); margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; }
            .header { background: var(--chat-bg); padding: 15px 25px; border-bottom: 1px solid #e5e7eb; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); z-index: 10; }
            .header h3 { margin: 0; color: #1f2937; font-size: 1.2rem; }
            .btn-group { display: flex; gap: 10px; }
            .btn { background: #f3f4f6; color: #4b5563; border: 1px solid #d1d5db; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 500; transition: all 0.2s;}
            .btn:hover { background: #e5e7eb; }
            .btn-primary { background: var(--primary-color); color: white; border: none; }
            .btn-primary:hover { background: #0e8c6d; }
            #chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; max-width: 900px; margin: 0 auto; width: 100%; }
            .message { max-width: 80%; padding: 12px 18px; border-radius: 12px; line-height: 1.6; font-size: 15px; word-break: break-all; }
            .user { background-color: var(--primary-color); color: white; align-self: flex-end; border-bottom-right-radius: 4px; }
            .ai { background-color: var(--chat-bg); color: #1f2937; align-self: flex-start; border-bottom-left-radius: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); border: 1px solid #e5e7eb; }
            #input-wrapper { background: var(--chat-bg); border-top: 1px solid #e5e7eb; padding: 20px; }
            #input-container { max-width: 900px; margin: 0 auto; display: flex; gap: 10px; }
            input[type="text"] { flex: 1; padding: 12px 16px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 15px; outline: none; }
            input[type="text"]:focus { border-color: var(--primary-color); box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2); }
            input[type="file"] { display: none; }
            .ai pre { background: #1e1e1e; color: #d4d4d4; padding: 10px; border-radius: 6px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="header">
            <h3>🤖 终极版 Agentic RAG 知识库</h3>
            <div class="btn-group">
                <input type="file" id="file-upload" accept=".txt,.pdf" onchange="uploadFile()">
                <button class="btn btn-primary" onclick="document.getElementById('file-upload').click()">📄 上传知识文档</button>
                <button class="btn" onclick="clearChat()">🗑️ 清空记忆</button>
            </div>
        </div>
        <div id="chat-container"></div>
        <div id="input-wrapper">
            <div id="input-container">
                <input type="text" id="user-input" placeholder="可查天气、查时间，或查询私有文档..." onkeydown="if(event.keyCode==13) send()">
                <button class="btn btn-primary" style="padding: 0 24px;" onclick="send()">发送</button>
            </div>
        </div>

        <script>
            if (!localStorage.getItem('session_id')) {
                localStorage.setItem('session_id', 'sess_' + Math.random().toString(36).substr(2, 9));
            }
            const sessionId = localStorage.getItem('session_id');

            function addMessage(text, type) {
                const chatContainer = document.getElementById('chat-container');
                const msgDiv = document.createElement('div');
                msgDiv.className = `message ${type}`;
                msgDiv.innerHTML = type === 'ai' ? marked.parse(text) : text;
                chatContainer.appendChild(msgDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
                return msgDiv;
            }

            async function clearChat() {
                await fetch(`/clear?session_id=${sessionId}`);
                document.getElementById('chat-container').innerHTML = '';
                addMessage('✅ 云端数据库与 Redis 缓存已清空。', 'ai');
            }

            async function uploadFile() {
                const fileInput = document.getElementById('file-upload');
                const file = fileInput.files[0];
                if (!file) return;
                addMessage(`⏳ 正在向量化切块《${file.name}》...`, 'ai');
                const formData = new FormData();
                formData.append("file", file);
                try {
                    const response = await fetch("/upload", { method: "POST", body: formData });
                    const result = await response.json();
                    addMessage(`✅ ${result.message}`, 'ai');
                } catch (e) { addMessage(`❌ 上传失败`, 'ai'); }
                fileInput.value = ''; 
            }

            async function send() {
                const input = document.getElementById('user-input');
                const text = input.value.trim();
                if (!text) return;
                addMessage(text, 'user');
                input.value = '';
                
                const aiBubble = document.createElement('div');
                aiBubble.className = 'message ai';
                aiBubble.innerText = '🧠 智能体思考中...';
                document.getElementById('chat-container').appendChild(aiBubble);
                
                const response = await fetch(`/ask?question=${encodeURIComponent(text)}&session_id=${sessionId}`);
                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let fullAiText = '';
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    fullAiText += decoder.decode(value);
                    aiBubble.innerHTML = marked.parse(fullAiText);
                    document.getElementById('chat-container').scrollTop = document.getElementById('chat-container').scrollHeight;
                }
            }
            
            window.onload = () => {
                addMessage('你好！我是进阶版 **全栈 Agent 智能体**。我不但能检索文档，还能自动调用 API 查天气、查时间！试试问我：“北京今天天气如何？”', 'ai');
            };
        </script>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = ""
    if file.filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file.file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted: content += extracted + "\n"
    else:
        content = (await file.read()).decode("utf-8")

    if not content.strip(): return {"status": "error", "message": "文件为空"}

    chunk_size = 500
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return {"status": "success", "message": f"成功学习《{file.filename}》，已激活知识库检索工具！"}

@app.get("/ask")
async def ask_ai(question: str, session_id: str, db: Session = Depends(get_db)):
    # 1. 从 Redis 或 DB 读取历史记忆
    cache_key = f"chat_history:{session_id}"
    current_history = []
    if redis_client:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            current_history = json.loads(cached_data)

    if not current_history:
        rows = db.query(History).filter(History.session_id == session_id).order_by(History.id.asc()).all()
        current_history = [{"role": row.role, "content": row.content} for row in rows]
        if redis_client and current_history:
            redis_client.setex(cache_key, 3600, json.dumps(current_history))

    # 2. 转换为 LangChain 消息格式
    lc_history = []
    for msg in current_history[-6:]: # 限制记忆长度防止超 token
        if msg["role"] == "user": lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant": lc_history.append(AIMessage(content=msg["content"]))

    # 3. 异步流式生成函数
    async def generate_stream():
        try:
            # yield "\n*[🧠 正在调度底层工具...]*\n\n"
            
            # 让大管家 Agent 自己去执行（它会自动决定是用天气工具、还是找知识库）
            # 注意：在企业应用中，AgentExecutor 自带的流式输出比较复杂，
            # 我们这里通过 invoke 拿到最终结果后，利用异步休眠模拟顺滑的打字机流式效果 (Fake Stream)
            response = agent_executor.invoke({
                "input": question,
                "chat_history": lc_history
            })
            
            final_reply = response["output"]

            # 平滑打字机输出特效
            chunk_size = 3
            for i in range(0, len(final_reply), chunk_size):
                await asyncio.sleep(0.01) # 毫秒级停顿
                yield final_reply[i:i+chunk_size]

            # 4. 对话结束后：异步更新数据库和 Redis (持久化记忆)
            with SessionLocal() as final_db:
                final_db.add(History(session_id=session_id, role="user", content=question))
                final_db.add(History(session_id=session_id, role="assistant", content=final_reply))
                final_db.commit()
            
            if redis_client:
                updated_history = current_history.copy()
                updated_history.append({"role": "user", "content": question})
                updated_history.append({"role": "assistant", "content": final_reply})
                redis_client.setex(cache_key, 3600, json.dumps(updated_history))

        except Exception as e:
            yield f"\n\n❌ AI 请求失败，请检查 API Key 配置: {str(e)}"

    return StreamingResponse(generate_stream(), media_type="text/event-stream; charset=utf-8")

@app.get("/clear")
def clear_history(session_id: str, db: Session = Depends(get_db)):
    db.query(History).filter(History.session_id == session_id).delete()
    db.commit()
    if redis_client: redis_client.delete(f"chat_history:{session_id}")
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860)
