import os
import requests
from dotenv import load_dotenv

# 1. 拿出抽屉里的钥匙（加载 .env 文件里的密钥）
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# 2. 确定收件地址和信封格式
url = "https://api.deepseek.com/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"  # 把钥匙塞进请求头里
}

# 3. 写信的内容（这是标准的通信格式，必须是字典嵌套列表）
data = {
    "model": "deepseek-chat", # 指定我们要用的模型名字
    "messages": [
        {"role": "system", "content": "你是一个幽默的编程导师。"},
        {"role": "user", "content": "你好！我刚刚跑通了第一段请求大模型的代码，快夸夸我！"}
    ],
    "temperature": 0.7  # 控制 AI 的发散程度，0.7 比较适中
}

# 4. 把信寄出去，并等待回信（这里用的是 requests 库自带的发送功能）
print("信件已寄出，等待 AI 回复中...")
response = requests.post(url, headers=headers, json=data)

# 5. 拆开回信，提取里面真正有用的文字
result = response.json()

# 我们加一个判断：如果回信里有 'choices'，说明成功；如果没有，说明报错了。
if 'choices' in result:
    ai_message = result['choices'][0]['message']['content']
    print("\n--- AI 的回信 ---")
    print(ai_message)
else:
    print("\n--- 🚨 服务器返回了错误通知 ---")
    print("原始信息：", result)