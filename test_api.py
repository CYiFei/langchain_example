from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 安全校验：检查密钥是否存在
if not os.getenv("DASHSCOPE_API_KEY"):
    raise ValueError(
        "❌ 未设置 DASHSCOPE_API_KEY！\n"
        "1. 请在项目根目录创建 .env 文件\n"
        "2. 添加内容: DASHSCOPE_API_KEY=sk-你的密钥\n"
        "3. 从 https://dashscope.console.aliyun.com/apiKey 获取密钥"
    )

# 额外验证：检查密钥格式（防无效密钥）
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key.startswith("sk-") or len(api_key) < 30:
    raise ValueError("❌ 无效的 DASHSCOPE_API_KEY 格式！密钥应以 'sk-' 开头")