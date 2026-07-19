from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import StreamingResponse, HTMLResponse
import os
import requests
import json
import chromadb
import PyPDF2
import uvicorn
from dotenv import load_dotenv

# 导入 SQLAlchemy 核心组件
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# 🔴 缓存架构 1：导入 Redis
import redis

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
load_dotenv()

app = FastAPI()
api_key = os.getenv("DEEPSEEK_API_KEY")

# ==========================================
# 数据库与缓存配置区
# ==========================================
# 1. 配置 PostgreSQL (永久存储)
raw_db_url = os.getenv("DATABASE_URL")
if raw_db_url and raw_db_url.startswith("postgres://"):
    database_url = raw_db_url.replace("postgres://", "postgresql://", 1)
else:
    database_url = raw_db_url

engine = create_engine(database_url)
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

# 🔴 缓存架构 2：初始化 Redis 连接 (热点内存)
redis_url = os.getenv("REDIS_URL")
# decode_responses=True 会自动把底层的字节码解码成字符串，极其方便
redis_client = redis.Redis.from_url(redis_url, decode_responses=True) if redis_url else None
if redis_client:
    print("✅ 成功连接到云端 Redis 内存加速引擎！")

# 向量数据库初始化
chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_or_create_collection(name="zhexin_kb")

@app.get("/", response_class=HTMLResponse)
async def home():
    # 前端 HTML 完全保持不变
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>哲鑫的全栈 RAG 知识库终端</title>
        <script src="https://cdn.bootcdn.net/ajax/libs/marked/4.2.12/marked.min.js"></script>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f7f7f8; margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; }
            .header { background: white; padding: 15px 20px; border-bottom: 1px solid #e5e7eb; display: flex; justify-content: space-between; align-items: center; }
            .header h3 { margin: 0; color: #374151; }
            .btn-group { display: flex; gap: 10px; }
            .btn { background: #f3f4f6; color: #4b5563; border: 1px solid #d1d5db; padding: 8px 15px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 500; }
            .btn:hover { background: #e5e7eb; }
            .btn-primary { background: #10a37f; color: white; border: none; }
            .btn-primary:hover { background: #0e8c6d; }
            #chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
            .message { max-width: 75%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; font-size: 15px; word-break: break-all; }
            .user { background-color: #10a37f; color: white; align-self: flex-end; border-bottom-right-radius: 2px; }
            .ai { background-color: white; color: #2d3748; align-self: flex-start; border-bottom-left-radius: 2px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
            #input-container { background: white; padding: 20px; display: flex; gap: 10px; border-top: 1px solid #e5e7eb; }
            input[type="text"] { flex: 1; padding: 14px; border: 1px solid #e5e7eb; border-radius: 8px; font-size: 15px; outline: none; }
            input[type="file"] { display: none; }
        </style>
    </head>
    <body>
        <div class="header">
            <h3>🤖 RAG 智能企业知识库 (哲鑫专属)</h3>
            <div class="btn-group">
                <input type="file" id="file-upload" accept=".txt,.pdf" onchange="uploadFile()">
                <button class="btn btn-primary" onclick="document.getElementById('file-upload').click()">+ 上传 PDF/TXT 喂养知识</button>
                <button class="btn" onclick="clearChat()">清空记忆</button>
            </div>
        </div>
        <div id="chat-container"></div>
        <div id="input-container">
            <input type="text" id="user-input" placeholder="针对已上传的文档提问..." onkeydown="if(event.keyCode==13) send()">
            <button class="btn btn-primary" style="padding: 0 24px;" onclick="send()">发送</button>
        </div>

        <script>
            if (!localStorage.getItem('session_id')) {
                localStorage.setItem('session_id', 'session_' + Math.random().toString(36).substr(2, 9));
            }
            const sessionId = localStorage.getItem('session_id');

            function addMessage(text, type) {
                const chatContainer = document.getElementById('chat-container');
                chatContainer.innerHTML += `<div class="message ${type}">${text}</div>`;
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            async function clearChat() {
                await fetch(`/clear?session_id=${sessionId}`);
                document.getElementById('chat-container').innerHTML = '';
                addMessage('云端数据库记忆已清空，我们重新开始吧！', 'ai');
            }

            async function uploadFile() {
                const fileInput = document.getElementById('file-upload');
                const file = fileInput.files[0];
                if (!file) return;

                addMessage(`正在努力阅读并背诵文档：《${file.name}》... 这可能需要十几秒钟。`, 'ai');
                
                const formData = new FormData();
                formData.append("file", file);

                try {
                    const response = await fetch("/upload", { method: "POST", body: formData });
                    const result = await response.json();
                    addMessage(`✅ ${result.message}`, 'ai');
                } catch (e) {
                    addMessage(`❌ 文档上传失败: ${e}`, 'ai');
                }
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
                aiBubble.innerText = '正在检索知识库...';
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
                addMessage('你好！我现在拥有了 RAG 超能力和 Redis 缓存。请上传文档开始提问吧！', 'ai');
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

    return {"status": "success", "message": f"成功学习了《{file.filename}》，共 {len(chunks)} 个碎片入库！"}

@app.get("/ask")
def ask_ai(question: str, session_id: str, db: Session = Depends(get_db)):
    # 1. 向量检索 (RAG)
    results = collection.query(query_texts=[question], n_results=2)
    retrieved_context = "\n\n---\n\n".join(results['documents'][0]) if results['documents'] and results['documents'][0] else ""
    
    # 🔴 架构优化：RAG 幻觉拦截（Prompt Engineering 护栏）
    # 给大模型增加一个判断指令，防止它在闲聊时强行引用毫无关联的背景知识
    if retrieved_context:
        enhanced_prompt = (
            f"背景知识:\n{retrieved_context}\n\n"
            f"问题: {question}\n\n"
            f"⚠️ 注意：请判断上述【背景知识】是否与【问题】真正相关。如果无关（例如用户只是在打招呼或闲聊），请完全忽略背景知识，像平时一样正常回答，绝不要提及背景知识的内容。"
        )
    else:
        enhanced_prompt = question

    # 🔴 缓存架构 3：优先从 Redis 内存中找历史记录（闪电般的速度）
    cache_key = f"chat_history:{session_id}"
    current_history = []
    
    if redis_client:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            current_history = json.loads(cached_data)
            print(f"⚡ [缓存命中 Cache Hit!] 极速读取内存，跳过数据库！Session: {session_id}")

    # 🔴 缓存架构 4：如果内存里没有 (Cache Miss)，再去查 Postgres 硬盘
    if not current_history:
        print(f"🐢 [缓存未命中 Cache Miss...] 老老实实去查 PostgreSQL 硬盘。Session: {session_id}")
        rows = db.query(History).filter(History.session_id == session_id).order_by(History.id.asc()).all()
        current_history = [{"role": row.role, "content": row.content} for row in rows]
        
        if not current_history:
            sys_msg = {"role": "system", "content": "你是一个专业的AI助手，精通Markdown排版。"}
            current_history.append(sys_msg)
            new_sys_record = History(session_id=session_id, role="system", content=sys_msg["content"])
            db.add(new_sys_record)
            db.commit()
            
        # 既然从硬盘查出来了，就把它放到桌子（Redis）上，并设置 1 小时后过期 (3600秒)
        if redis_client:
            redis_client.setex(cache_key, 3600, json.dumps(current_history))

    api_messages = current_history.copy()
    api_messages.append({"role": "user", "content": enhanced_prompt})

    # 保存新的用户提问到硬盘
    new_user_record = History(session_id=session_id, role="user", content=question)
    db.add(new_user_record)
    db.commit()
    
    def generate_stream():
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        data = {"model": "deepseek-chat", "messages": api_messages, "stream": True}
        
        try:
            response = requests.post(url, headers=headers, json=data, stream=True)
            full_reply = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: ") and "[DONE]" not in decoded_line:
                        try:
                            chunk = json.loads(decoded_line[6:])
                            text = chunk['choices'][0]['delta'].get('content', '')
                            if text:
                                full_reply += text
                                yield text
                        except: pass
            
            if full_reply:
                with SessionLocal() as final_db:
                     new_ai_record = History(session_id=session_id, role="assistant", content=full_reply)
                     final_db.add(new_ai_record)
                     final_db.commit()
                
                # 🔴 缓存架构 5：缓存平滑更新 (Cache Update)
                # 不再暴力清空桌子！而是把最新的对话（用户提问 + AI回答）追加到缓存列表中，
                # 这样下一次用户再提问时，就能真正触发闪电般的 ⚡ [缓存命中] 了！
                if redis_client:
                    updated_history = current_history.copy()
                    updated_history.append({"role": "user", "content": question})
                    updated_history.append({"role": "assistant", "content": full_reply})
                    redis_client.setex(cache_key, 3600, json.dumps(updated_history))

        except Exception as e:
            yield f"网络断开: {e}"

    return StreamingResponse(generate_stream(), media_type="text/event-stream; charset=utf-8")

@app.get("/clear")
def clear_history(session_id: str, db: Session = Depends(get_db)):
    db.query(History).filter(History.session_id == session_id).delete()
    db.commit()
    # 清空硬盘的同时，也要把内存里的缓存删掉
    if redis_client:
        redis_client.delete(f"chat_history:{session_id}")
    return {"status": "success"}

if __name__ == "__main__":
    # 一键启动入口：直接点击 VS Code 右上角的运行按钮即可启动服务
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)