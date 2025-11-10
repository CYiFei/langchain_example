import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi  # 通义千问专用类
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# 加载环境变量（已通过验证脚本确认）
load_dotenv()

async def main():
    model = ChatTongyi(model="qwen3-max", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
    parser = StrOutputParser()
    chain = prompt | model | parser

    async for event in chain.astream_events({"topic": "parrot"}, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            print(event, end="|", flush=True)

# 运行异步函数
if __name__ == "__main__":
    asyncio.run(main())