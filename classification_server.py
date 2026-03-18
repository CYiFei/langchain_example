import os
import time
import logging
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from langchain_community.chat_models import ChatTongyi
from fastapi import FastAPI, HTTPException
from uvicorn import run

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class TextClassificationResult(BaseModel):
    category: str = Field(description="文本所属的分类标签")
    confidence: float = Field(description="分类置信度（0-1 之间）")
    reasoning: str = Field(description="分类的理由说明")

    @field_validator('confidence')
    @classmethod
    def check_confidence_range(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("置信度必须在 0 到 1 之间")
        return v

class ClassificationRequest(BaseModel):
    text: str
    classification_type: str
    categories: list[str]

def init_qwen_model():
    model = ChatTongyi(
        model_name="qwen3-max",
        temperature=0.1,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        max_tokens=1024
    )
    return model

def build_classification_prompt(classification_type, categories):
    parser = PydanticOutputParser(pydantic_object=TextClassificationResult)
    
    prompt_template = PromptTemplate(
        template="""
你是一个专业的{classification_type}专家，需要按照以下要求对输入文本进行分类：

1. 可选分类标签：{categories}
2. 必须从指定标签中选择一个，禁止自定义标签
3. 给出 0-1 之间的置信度，表示分类结果的确定程度
4. 详细说明分类的理由，需基于文本内容客观分析

输入文本：{text}

{format_instructions}
        """,
        input_variables=["classification_type", "categories", "text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return prompt_template, parser

def build_classification_chain(classification_type, categories):
    model = init_qwen_model()
    prompt, parser = build_classification_prompt(classification_type, categories)
    classification_chain = prompt | model | parser
    return classification_chain

app = FastAPI(title="文本分类服务", description="基于 LangChain + Qwen 的文本分类 API")

@app.post("/classify", response_model=TextClassificationResult)
async def classify_text(request: ClassificationRequest):
    request_start_time = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"收到分类请求 - ID: {id(request)}")
    logger.info(f"文本：{request.text[:50]}..." if len(request.text) > 50 else f"文本：{request.text}")
    logger.info(f"分类类型：{request.classification_type}")
    logger.info(f"分类标签：{request.categories}")
    
    try:
        chain_build_start = time.time()
        chain = build_classification_chain(request.classification_type, request.categories)
        chain_build_time = time.time() - chain_build_start
        logger.info(f"链构建耗时：{chain_build_time:.4f}秒")
        
        invoke_start = time.time()
        result = chain.invoke({
            "text": request.text,
            "classification_type": request.classification_type,
            "categories": request.categories
        })
        invoke_time = time.time() - invoke_start
        
        total_time = time.time() - request_start_time
        
        logger.info(f"\n模型推理完成:")
        logger.info(f"  分类结果：{result.category}")
        logger.info(f"  置信度：{result.confidence:.4f}")
        logger.info(f"  推理耗时：{invoke_time:.4f}秒")
        logger.info(f"  总耗时：{total_time:.4f}秒")
        logger.info(f"{'='*60}\n")
        
        return result
    except Exception as e:
        total_time = time.time() - request_start_time
        logger.error(f"请求处理失败 - 耗时：{total_time:.4f}秒")
        logger.error(f"错误信息：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("===== 情感分类示例 =====")
    sentiment_chain = build_classification_chain(
        classification_type="情感分类",
        categories=["正面", "负面", "中性"]
    )
    sentiment_result = sentiment_chain.invoke({
        "text": "这款手机续航特别好，用了一天还有 50% 的电，就是系统偶尔有点卡",
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
        "text": "我的订单已经付款 3 天了还没发货，能帮我查一下原因吗？",
        "classification_type": "用户意图分类",
        "categories": ["咨询产品", "投诉问题", "寻求帮助", "建议反馈"]
    })
    print(f"分类结果：{intent_result.category}")
    print(f"置信度：{intent_result.confidence}")
    print(f"分类理由：{intent_result.reasoning}")