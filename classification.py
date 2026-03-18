import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from langchain_community.chat_models import ChatTongyi  # ✅ 官方支持

# 加载环境变量
load_dotenv()

# ====================== 1. 定义分类数据结构 ======================
class TextClassificationResult(BaseModel):
    category: str = Field(description="文本所属的分类标签")
    confidence: float = Field(description="分类置信度（0-1之间）")
    reasoning: str = Field(description="分类的理由说明")

    @field_validator('confidence')
    @classmethod
    def check_confidence_range(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("置信度必须在 0 到 1 之间")
        return v

# ====================== 2. 初始化Qwen模型 ======================
def init_qwen_model():
    model = ChatTongyi(
        model_name="qwen3-max",  # 或 qwen-plus, qwen-turbo；qwen3.5-plus 可能无效
        temperature=0.1,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        max_tokens=1024
    )
    return model

# ====================== 3. 构建分类提示词模板 ======================
def build_classification_prompt(classification_type, categories):
    parser = PydanticOutputParser(pydantic_object=TextClassificationResult)
    
    prompt_template = PromptTemplate(
        template="""
你是一个专业的{classification_type}专家，需要按照以下要求对输入文本进行分类：

1. 可选分类标签：{categories}
2. 必须从指定标签中选择一个，禁止自定义标签
3. 给出0-1之间的置信度，表示分类结果的确定程度
4. 详细说明分类的理由，需基于文本内容客观分析

输入文本：{text}

{format_instructions}
        """,
        input_variables=["classification_type", "categories", "text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return prompt_template, parser

# ====================== 4. 构建分类工作流 ======================
def build_classification_chain(classification_type, categories):
    model = init_qwen_model()
    prompt, parser = build_classification_prompt(classification_type, categories)
    classification_chain = prompt | model | parser
    return classification_chain

# ====================== 5. 测试示例 ======================
if __name__ == "__main__":
    print("===== 情感分类示例 =====")
    sentiment_chain = build_classification_chain(
        classification_type="情感分类",
        categories=["正面", "负面", "中性"]
    )
    sentiment_result = sentiment_chain.invoke({
        "text": "这款手机续航特别好，用了一天还有50%的电，就是系统偶尔有点卡",
        "classification_type": "情感分类",
        "categories": ["正面", "负面", "中性"]
    })
    print(f"分类结果：{sentiment_result.category}")
    print(f"置信度：{sentiment_result.confidence}")
    print(f"分类理由：{sentiment_result.reasoning}\n")

    print("===== 意图分类示例 =====")
    intent_chain = build_classification_chain(
        classification_type="用户意图分类",
        categories=["咨询产品", "投诉问题", "寻求帮助", "建议反馈"]
    )
    intent_result = intent_chain.invoke({
        "text": "我的订单已经付款3天了还没发货，能帮我查一下原因吗？",
        "classification_type": "用户意图分类",
        "categories": ["咨询产品", "投诉问题", "寻求帮助", "建议反馈"]
    })
    print(f"分类结果：{intent_result.category}")
    print(f"置信度：{intent_result.confidence}")
    print(f"分类理由：{intent_result.reasoning}")