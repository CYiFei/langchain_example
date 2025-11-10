from typing import Optional

from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatTongyi  # 通义千问专用类

from dotenv import load_dotenv

# 加载环境变量（已通过验证脚本确认）
load_dotenv()

# ✅ 关键修复：指定 qwen-turbo（免费额度充足，无需额外开通）
llm = ChatTongyi(model="qwen3-max", temperature=0.7)

class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")

structured_llm = llm.with_structured_output(Joke)

structured_llm.invoke("Tell me a joke about cats")