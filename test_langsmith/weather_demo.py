# weather_demo.py
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatTongyi

# 加载 .env 文件
load_dotenv()
print("LangSmith Project:", os.getenv("LANGCHAIN_PROJECT"))
print("DashScope Key set:", bool(os.getenv("DASHSCOPE_API_KEY")))
# === 工具定义 ===
@tool
def get_weather(location: str) -> str:
    """获取指定城市的当前天气（模拟数据）"""
    location = location.strip()
    if "北京" in location:
        return "北京当前天气晴，气温7°C，空气质量良。"
    elif "上海" in location:
        return "上海多云，气温12°C，微风。"
    elif "广州" in location:
        return "广州阴天，气温18°C，湿度较高。"
    else:
        return f"抱歉，暂时无法获取 {location} 的天气信息。"

tools = [get_weather]

# === 初始化 Qwen3-Max 模型 ===
llm = ChatTongyi(
    model="qwen-max",  # Qwen3-Max 在 DashScope 中的模型名仍是 qwen-max
    temperature=0,
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    streaming=True
)

# === 构建 Prompt ===
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的天气助手，请使用工具查询用户指定城市的天气，并给出简洁清晰的回答。"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

# === 创建 Agent ===
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# === 执行 Demo ===
if __name__ == "__main__":
    question = "北京今天天气怎么样？"
    print(f"👤 用户提问: {question}\n")

    try:
        result = agent_executor.invoke({"input": question})
        print("\n🤖 最终回答:", result["output"])
    except Exception as e:
        print("❌ 执行出错:", str(e))