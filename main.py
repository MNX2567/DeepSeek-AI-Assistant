from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import StreamingResponse, HTMLResponse
import os
import requests
import json
import chromadb
import PyPDF2
import uvicorn
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import redis

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
load_dotenv()

app = FastAPI()
api_key = os.getenv("DEEPSEEK_API_KEY")

# 数据库配置
raw_db_url = os.getenv("DATABASE_URL")
database_url = raw_db_url.replace("postgres://", "postgresql://", 1) if raw_db_url and raw_db_url.startswith("postgres://") else raw_db_url
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

redis_url = os.getenv("REDIS_URL")
redis_client = redis.Redis.from_url(redis_url, decode_responses=True) if redis_url else None
chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_or_create_collection(name="zhexin_kb")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <body><h1>RAG 服务已启动</h1></body>
    </html>
    """

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8") if not file.filename.endswith(".pdf") else ""
    if file.filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file.file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted: content += extracted + "\n"
    
    if not content.strip(): return {"status": "error", "message": "文件为空"}
    chunk_size = 500
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return {"status": "success", "message": f"学习成功！"}

@app.get("/ask")
def ask_ai(question: str, session_id: str, db: Session = Depends(get_db)):
    results = collection.query(query_texts=[question], n_results=2)
    retrieved_context = "\n\n---\n\n".join(results['documents'][0]) if results['documents'] and results['documents'][0] else ""
    enhanced_prompt = f"背景知识:\n{retrieved_context}\n\n问题: {question}" if retrieved_context else question

    # 获取历史记录
    rows = db.query(History).filter(History.session_id == session_id).order_by(History.id.asc()).all()
    api_messages = [{"role": row.role, "content": row.content} for row in rows]
    api_messages.append({"role": "user", "content": enhanced_prompt})

    def generate_stream():
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        data = {"model": "deepseek-chat", "messages": api_messages, "stream": True}
        response = requests.post(url, headers=headers, json=data, stream=True)
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("data: ") and "[DONE]" not in decoded:
                    try:
                        chunk = json.loads(decoded[6:])
                        text = chunk['choices'][0]['delta'].get('content', '')
                        if text: yield text
                    except: pass
    return StreamingResponse(generate_stream(), media_type="text/event-stream; charset=utf-8")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860)
