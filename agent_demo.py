
# 运行前需要安装库：
# pip install langchain langchain-openai

import os
from dotenv import load_dotenv

## 核心组件导入
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 这一行是万能的，它直接从核心定义文件中导入 AgentExecutor
from langchain.agents.agent import AgentExecutor
from langchain.agents import create_tool_calling_agent
# 加载 .env 环境变量中的 DEEPSEEK_API_KEY
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# ==========================================
# 第一步：打造 AI 的“手”和“脚” (定义工具 Tools)
# ==========================================

# 加上 @tool 装饰器，并且一定要写清晰的函数注释（Docstring），
# 因为大模型是靠读你的注释，来决定什么时候使用这个工具的！
@tool
def get_weather(city: str) -> str:
    """当你需要查询任何城市的天气时，调用此工具。传入参数为城市名称。"""
    print(f"\n[🛠️ 系统底层运行] 大模型指令到达！正在调用外部工具查询 {city} 的天气...")
    
    # 在真实企业项目中，这里会调用真实的第三方天气 API
    # 为了演示，我们在这里做个简单的模拟返回
    mock_weather_db = {
        "北京": "晴天，气温 25℃，空气质量优",
        "上海": "暴雨，气温 18℃，建议带伞",
        "太原": "多云，气温 22℃，微风"
    }
    
    return mock_weather_db.get(city, f"抱歉，未能在气象局数据库中查找到 {city} 的天气。")

@tool
def get_express_status(tracking_number: str) -> str:
    """当用户查询快递状态、物流信息时，调用此工具。传入参数为快递单号。"""
    print(f"\n[🛠️ 系统底层运行] 正在连接顺丰/中通数据库，查询单号 {tracking_number} ...")
    return f"单号 {tracking_number} 的包裹已到达【太原理工大学菜鸟驿站】，请凭取件码 88-5-1200 提取。"

# 我们把造好的工具放进一个列表里
tools = [get_weather, get_express_status]


# ==========================================
# 第二步：唤醒 AI 大脑 (初始化 LLM)
# ==========================================
# DeepSeek 兼容 OpenAI 的接口格式，所以直接用 ChatOpenAI 积木
llm = ChatOpenAI(
    model="deepseek-chat", 
    api_key=api_key, 
    base_url="https://api.deepseek.com",
    max_tokens=1024
)


# ==========================================
# 第三步：设定规矩 (编写 Prompt 提示词)
# ==========================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个万能的 AI 智能体（Agent）。你可以使用提供的各种工具来帮助用户解决问题。"),
    ("human", "{input}"),
    # 下面这行很重要，它是 AI 思考和记录调用工具过程的“草稿本”
    ("placeholder", "{agent_scratchpad}"),
])


# ==========================================
# 第四步：组装智能体大管家 (Agent Executor)
# ==========================================
# 把大脑(llm)、工具(tools)和规矩(prompt)拼装在一起
agent = create_tool_calling_agent(llm, tools, prompt)

# 开启 verbose=True，你可以像看黑客帝国一样，在终端看到 AI 的心智思考过程！
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# ==========================================
# 第五步：见证奇迹的测试环节
# ==========================================
if __name__ == "__main__":
    print("🤖 AI Agent 已启动！")
    
    # 测试一：普通的闲聊，AI 应该直接回答，不需要用工具
    print("\n\n>>> 用户提问 1: 你好，你是谁？")
    agent_executor.invoke({"input": "你好，你是谁？"})
    
    # 测试二：触发天气工具
    print("\n\n>>> 用户提问 2: 今天太原的天气怎么样？出门需要带伞吗？")
    agent_executor.invoke({"input": "今天太原的天气怎么样？出门需要带伞吗？"})
    
    # 测试三：多重逻辑（AI 会聪明地先查天气，再做决定）
    print("\n\n>>> 用户提问 3: 我的快递单号是 SF123456。对了，顺便告诉我上海今天天气，我寄快递过去怕淋湿。")
    agent_executor.invoke({"input": "我的快递单号是 SF123456。对了，顺便告诉我上海今天天气，我寄快递过去怕淋湿。"})