from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

app = FastAPI(
    title="LangChain Server with Qwen3-Max",
    version="1.0",
    description="A LangChain server using Qwen3-Max model via DashScope API",
)


@app.get("/")
async def redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse("/docs")

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的AI助手，使用中文回答用户问题。"),
    ("human", "{input}")
])

# 初始化 Qwen3-Max 模型
qwen_model = ChatTongyi(
    model_name="qwen3-max",
    temperature=0.7,
    max_tokens=2000,
)

# 创建处理链
chain = (
    {"input": RunnablePassthrough()} 
    | prompt 
    | qwen_model 
    | StrOutputParser()
)

# 添加路由
add_routes(app, chain, path="/chat")
add_routes(app, prompt | qwen_model, path="/chat/raw")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)