import requests
import json
import argparse
from typing import Dict, Any, Optional

class QwenClient:
    """客户端用于调用基于Qwen3-Max的LangChain服务"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化客户端
        
        Args:
            base_url: 服务器基础URL，默认为本地开发地址
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def chat(self, message: str, raw: bool = False) -> Dict[str, Any]:
        """
        与Qwen3-Max模型对话
        
        Args:
            message: 用户输入的消息
            raw: 是否使用/raw端点获取原始响应
        
        Returns:
            模型响应的字典格式
        """
        endpoint = "/chat/raw/invoke" if raw else "/chat/invoke"
        url = f"{self.base_url}{endpoint}"
        
        # 修复：确保发送正确的JSON格式
        payload = {"input": message}
        print(f"➡️ 发送请求到 {url}，内容: {json.dumps(payload, ensure_ascii=False)}")
        try:
            # 修复：使用正确的请求方式
            response = self.session.post(
                url,
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # 详细打印错误信息
            error_detail = "No detail available"
            if e.response:
                try:
                    error_detail = e.response.json()
                except:
                    error_detail = e.response.text
            return {
                "error": str(e),
                "status_code": e.response.status_code if e.response else None,
                "detail": error_detail
            }
    
    def pretty_print_response(self, response: Dict[str, Any], raw: bool = False) -> None:
        """
        美化打印响应结果
        
        Args:
            response: API响应
            raw: 是否是原始响应
        """
        print("\n" + "="*50)
        
        if "error" in response:
            print(f"❌ 错误: {response['error']}")
            if response.get("status_code"):
                print(f"状态码: {response['status_code']}")
            if "detail" in response:
                print(f"错误详情: {json.dumps(response['detail'], indent=2, ensure_ascii=False)}")
            return
        
        if raw:
            # 处理原始响应格式
            print("🤖 Qwen3-Max (原始响应):")
            if "output" in response:
                if isinstance(response["output"], dict) and "content" in response["output"]:
                    print(f"回答: {response['output']['content']}")
                else:
                    print(f"完整响应: {json.dumps(response['output'], indent=2, ensure_ascii=False)}")
            else:
                print(f"响应: {json.dumps(response, indent=2, ensure_ascii=False)}")
        else:
            # 处理纯文本响应
            print("🤖 Qwen3-Max (简洁响应):")
            if "output" in response:
                print(f"回答: {response['output']}")
            else:
                print(f"响应: {json.dumps(response, indent=2, ensure_ascii=False)}")
        
        print("="*50 + "\n")

def interactive_mode(client: QwenClient, raw: bool = False) -> None:
    """交互式对话模式"""
    print("\n🚀 已进入Qwen3-Max对话模式 (输入'exit'或'quit'退出)")
    print(f"   使用{'原始' if raw else '简洁'}响应格式\n")
    
    while True:
        try:
            user_input = input("👤 您说: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 再见！\n")
                break
            
            if not user_input:
                continue
                
            response = client.chat(user_input, raw)
            client.pretty_print_response(response, raw)
            
        except KeyboardInterrupt:
            print("\n\n👋 通过键盘中断退出\n")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}\n")

def single_query(client: QwenClient, query: str, raw: bool = False) -> None:
    """单次查询模式"""
    print(f"\n🔍 查询: {query}")
    response = client.chat(query, raw)
    client.pretty_print_response(response, raw)

def test_connection(client: QwenClient) -> bool:
    """测试与服务器的连接"""
    try:
        response = client.session.get(f"{client.base_url}/docs")
        if response.status_code == 200:
            print("✅ 成功连接到服务器")
            return True
        else:
            print(f"❌ 服务器返回状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到服务器: {str(e)}")
        print(f"   请确保服务器正在运行: {client.base_url}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Qwen3-Max API 客户端')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                        help='服务器URL (默认: http://localhost:8000)')
    parser.add_argument('--query', type=str, 
                        help='单次查询内容，如果不提供则进入交互模式')
    parser.add_argument('--raw', action='store_true',
                        help='使用原始响应格式 (/chat/raw 端点)')
    parser.add_argument('--test', action='store_true',
                        help='仅测试服务器连接')
    
    args = parser.parse_args()
    
    client = QwenClient(args.url)
    
    # 先测试连接
    if not test_connection(client):
        return
    
    if args.test:
        return
    
    if args.query:
        # 单次查询模式
        single_query(client, args.query, args.raw)
    else:
        # 交互模式
        interactive_mode(client, args.raw)

if __name__ == "__main__":
    print("✨ Qwen3-Max LangChain 客户端 ✨")
    print("   使用 --help 查看可用选项\n")
    main()