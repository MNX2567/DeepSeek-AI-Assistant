from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import os
import requests
import json
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

chat_sessions = {}

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>哲鑫的生产级 AI 助手 (企业级防刷版)</title>
        <script src="https://cdn.bootcdn.net/ajax/libs/marked/4.2.12/marked.min.js"></script>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f7f7f8; margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; }
            .header { background: white; padding: 15px 20px; border-bottom: 1px solid #e5e7eb; display: flex; justify-content: space-between; align-items: center; }
            .header h3 { margin: 0; color: #374151; }
            .clear-btn { background: #f3f4f6; color: #4b5563; border: 1px solid #d1d5db; padding: 8px 15px; border-radius: 6px; cursor: pointer; font-size: 14px; }
            .clear-btn:hover { background: #e5e7eb; }
            #chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
            .message { max-width: 75%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; font-size: 15px; word-break: break-all; }
            .user { background-color: #10a37f; color: white; align-self: flex-end; border-bottom-right-radius: 2px; }
            .ai { background-color: white; color: #2d3748; align-self: flex-start; border-bottom-left-radius: 2px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
            .ai pre { background: #f4f4f4; padding: 10px; border-radius: 6px; overflow-x: auto; }
            .ai code { font-family: monospace; background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }
            #input-container { background: white; padding: 20px; display: flex; gap: 10px; border-top: 1px solid #e5e7eb; }
            input { flex: 1; padding: 14px; border: 1px solid #e5e7eb; border-radius: 8px; font-size: 15px; outline: none; }
            input:focus { border-color: #10a37f; }
            button.send-btn { background-color: #10a37f; color: white; border: none; padding: 0 24px; border-radius: 8px; font-size: 15px; cursor: pointer; font-weight: bold; }
            button.send-btn:hover { background-color: #0e8c6d; }
        </style>
    </head>
    <body>
        <div class="header">
            <h3>🤖 DeepSeek 智能生产终端</h3>
            <button class="clear-btn" onclick="clearChat()">+ 新对话</button>
        </div>
        <div id="chat-container"></div>
        <div id="input-container">
            <input type="text" id="user-input" placeholder="给 AI 发送消息..." onkeydown="if(event.keyCode==13) send()">
            <button class="send-btn" onclick="send()">发送</button>
        </div>

        <script>
            if (!localStorage.getItem('session_id')) {
                localStorage.setItem('session_id', 'session_' + Math.random().toString(36).substr(2, 9));
            }
            const sessionId = localStorage.getItem('session_id');

            async function clearChat() {
                await fetch(`/clear?session_id=${sessionId}`);
                document.getElementById('chat-container').innerHTML = '<div class="message ai">记忆已清空，我们重新开始吧！</div>';
            }

            async function send() {
                const input = document.getElementById('user-input');
                const text = input.value.trim();
                if (!text) return;
                
                const chatContainer = document.getElementById('chat-container');
                chatContainer.innerHTML += `<div class="message user">${text}</div>`;
                input.value = '';
                chatContainer.scrollTop = chatContainer.scrollHeight;
                
                const aiBubble = document.createElement('div');
                aiBubble.className = 'message ai';
                aiBubble.innerText = '正在思考...';
                chatContainer.appendChild(aiBubble);
                
                const response = await fetch(`/ask?question=${encodeURIComponent(text)}&session_id=${sessionId}`);
                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                
                let fullAiText = '';
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    fullAiText += chunk;
                    
                    // 🔴 改变 2：企业级容错护盾！如果 marked 库加载失败，自动降级为纯文本，绝不卡死
                    if (typeof marked !== 'undefined') {
                        aiBubble.innerHTML = marked.parse(fullAiText);
                    } else {
                        aiBubble.innerText = fullAiText; 
                    }
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            }
            
            window.onload = () => {
                document.getElementById('chat-container').innerHTML = '<div class="message ai">你好！我是你的专属 AI 助手，今天想聊点什么？</div>';
            };
        </script>
    </body>
    </html>
    """

@app.get("/ask")
def ask_ai(question: str, session_id: str):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = [{"role": "system", "content": "你是一个专业的AI助手，回答时尽量使用 Markdown 格式让排版更好看。"}]
    
    current_history = chat_sessions[session_id]
    current_history.append({"role": "user", "content": question})
    
    def generate_stream():
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": "deepseek-chat",
            "messages": current_history,
            "stream": True
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, stream=True)
            full_reply = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[6:]
                        if json_str == "[DONE]":
                            current_history.append({"role": "assistant", "content": full_reply})
                            break
                        try:
                            chunk = json.loads(json_str)
                            text = chunk['choices'][0]['delta'].get('content', '')
                            if text:
                                full_reply += text
                                # 🔴 改变 3：把后端的强制冲刷请回来，双重保险防止操作系统缓存截留
                                print(text, end="", flush=True)
                                yield text
                        except:
                            pass
        except Exception as e:
            yield "网络断开啦"

    return StreamingResponse(generate_stream(), media_type="text/event-stream; charset=utf-8")

@app.get("/clear")
def clear_history(session_id: str):
    if session_id in chat_sessions:
        chat_sessions.pop(session_id)
    return {"status": "success"}