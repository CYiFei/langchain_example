from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi  # 通义千问专用类
import os
from dotenv import load_dotenv

# 加载环境变量（已通过验证脚本确认）
load_dotenv()

# ✅ 关键修复：指定 qwen-turbo（免费额度充足，无需额外开通）
llm = ChatTongyi(model="qwen3-max", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是世界级的技术专家"),
    ("user", "{input}")
])

chain = prompt | llm

# 调用并提取内容
result = chain.invoke({"input": "帮我写一篇关于AI的技术文章，100个字"})
print(result.content)  # 正确输出文本内容