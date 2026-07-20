import os
import json
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import PyPDF2
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_or_create_collection(name="zhexin_kb")

@tool
def search_knowledge_base(query: str) -> str:
    """当用户提问关于上传的私有文档、简历或本地知识库的内容时，必须调用此工具进行检索。"""
    print(f"\n[🛠️ 工具调用] 正在检索本地知识库，关键词：{query}")
    results = collection.query(query_texts=[query], n_results=2)
    if results['documents'] and results['documents'][0]:
         return "\n\n".join(results['documents'][0])
    return "知识库中未找到相关内容。"

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    print(f"\n[🛠️ 工具调用] 正在查询 {city} 的天气...")
    return f"{city}的天气是多云转晴，25℃，空气质量优，适合出行。"

import datetime

@tool
def get_current_time() -> str:
    """当用户询问现在的时间、今天几号、今天是星期几等关于当前时间的问题时，必须调用此工具。"""
    now = datetime.datetime.now()
    # 格式化时间，让 AI 读起来更清晰
    time_str = now.strftime("%Y年%m月%d日 %H时%M分")
    print(f"\n[🛠️ 工具调用] 正在获取系统当前时间：{time_str}")
    return f"现在的系统时间是：{time_str}。"

# 将新工具放入工具箱
tools = [search_knowledge_base, get_weather, get_current_time]

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个万能的全栈AI助理。你可以通过工具查询天气，也可以通过工具检索私有知识库。如果用户的问题不需要工具，请直接友善地回答。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """文件上传与向量化 (RAG核心)"""
    content = ""
    if file.filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file.file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted: content += extracted + "\n"
    else:
        content = (await file.read()).decode("utf-8")

    if not content.strip(): 
        return {"status": "error", "message": "文件为空"}

    # 简单的文本切块
    chunk_size = 500
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return {"status": "success", "message": f"成功学习了《{file.filename}》！你可以通过提问来检索它了。"}

@app.get("/chat")
async def chat_endpoint(question: str, session_id: str = "default_session"):
    """对话接口，直接调用 Agent 处理"""
    
    # 将输出包装成异步生成器，适配前端的流式读取
    async def generate():
        try:
            # 把问题交给大管家 Agent，它会自动决定是用天气工具、检索工具，还是直接回答
            response = agent_executor.invoke({"input": question})
            yield response["output"]
        except Exception as e:
            yield f"AI 思考时发生了错误: {str(e)}"

    return StreamingResponse(generate(), media_type="text/event-stream; charset=utf-8")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)