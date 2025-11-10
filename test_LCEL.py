import os
from dotenv import load_dotenv
# from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI

# 从 .env 文件加载环境变量
load_dotenv()

# 读取 Azure OpenAI 配置
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_deployment = os.getenv("AZURE_DEPLOYMENT_NAME")

if not all([azure_api_key, azure_endpoint, azure_api_version, azure_deployment]):
    raise ValueError("缺少必要的环境变量，请检查 .env 文件是否正确配置。")

# 创建 AzureChatOpenAI 模型实例
model = AzureChatOpenAI(
    azure_deployment=azure_deployment,      # 部署名称
    api_version=azure_api_version,          # API 版本
    azure_endpoint=azure_endpoint,          # endpoint URL
    api_key=azure_api_key,                  # API Key
    temperature=0.7,
)

# 可选：测试调用
from langchain_core.messages import HumanMessage
result = model.invoke([HumanMessage(content="你好，世界！")])
print(result.content)

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

# model.invoke(messages)
result = model.invoke(messages)
# print("完整对象：", result)
# print("仅内容：", result.content)

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

result = model.invoke(messages)
print(parser.invoke(result))