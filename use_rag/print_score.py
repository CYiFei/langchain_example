# main_rag.py
import os
import sys
import time
import textwrap
import json
import traceback
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# 加载环境变量
load_dotenv()

# 设置 USER_AGENT (解决警告)
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "rag-application/1.0")

# 新式 LangChain 导入
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

class RAGSystem:
    """现代化的 RAG 系统，修复了字典输入问题"""
    
    def __init__(self, documents_dir: str = "documents", index_dir: str = "faiss_index"):
        self.documents_dir = documents_dir
        self.index_dir = index_dir
        self.embeddings = None
        self.vector_db = None
        self.llm = None
        self.rag_chain = None
        self.retriever = None
        self.chat_history = {}
        self.session_id = f"session_{int(time.time())}"
        
        # 配置国内镜像
        self._configure_mirrors()
    
    def _configure_mirrors(self):
        """配置国内镜像源加速下载"""
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("✅ 已配置 HuggingFace 国内镜像源")
    
    def load_documents(self) -> List[Document]:
        """加载多种格式的文档"""
        if not os.path.exists(self.documents_dir):
            os.makedirs(self.documents_dir)
            print(f"📁 创建文档目录: {self.documents_dir}")
            print("💡 请将 PDF、TXT 或 DOCX 文件放入此目录，然后重新运行程序")
            return []
        
        print(f"📚 开始加载文档，目录: {self.documents_dir}")
        
        # 配置加载器
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": lambda f: TextLoader(f, encoding='utf-8'),
            ".docx": Docx2txtLoader
        }
        
        documents = []
        processed_files = 0
        skipped_files = 0
        
        # 遍历目录中的所有文件
        for filename in os.listdir(self.documents_dir):
            filepath = os.path.join(self.documents_dir, filename)
            ext = os.path.splitext(filename)[1].lower()
            
            if not os.path.isfile(filepath):
                continue
            
            if ext in loaders:
                print(f"📄 正在处理: {filename}")
                try:
                    loader = loaders[ext](filepath)
                    docs = loader.load()
                    
                    # 添加元数据
                    for doc in docs:
                        doc.metadata.update({
                            "source": filename,
                            "file_path": filepath,
                            "file_type": ext,
                            "processed_time": time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                    
                    documents.extend(docs)
                    processed_files += 1
                except Exception as e:
                    print(f"❌ 处理文件 {filename} 时出错: {str(e)}")
                    print(f"堆栈跟踪: {traceback.format_exc()}")
                    skipped_files += 1
            else:
                print(f"⚠️  跳过不支持的文件类型: {filename}")
                skipped_files += 1
        
        print(f"✅ 文档加载完成: {processed_files} 个文件成功, {skipped_files} 个文件跳过")
        print(f"📊 总共加载了 {len(documents)} 个文档片段")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """智能分割文档"""
        if not documents:
            return []
        
        print("✂️  正在分割文档...")
        
        # 中文友好的分隔符
        separators = [
            "\n\n",  # 段落分隔
            "\n",    # 行分隔
            "。",    # 句号
            "；",    # 分号
            "？",    # 问号
            "！",    # 感叹号
            "……",   # 省略号
            "，",    # 逗号
            " ",     # 空格
            ""       # 字符级别
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=separators,
            keep_separator=True,
            is_separator_regex=False
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # 为每个文本块添加唯一ID
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"chunk_{i}"
            chunk.metadata["chunk_index"] = i
        
        print(f"✅ 文档分割完成: 生成了 {len(chunks)} 个文本块")
        # 打印分块内容
        self.show_document_chunks(chunks)
        return chunks
    
    def show_document_chunks(self, chunks: List[Document]):
        """显示所有文档分块内容"""
        if not chunks:
            print("⚠️  没有文档分块可显示")
            return
        
        print("\n" + "="*80)
        print(f"📋 详细文档分块内容 (共 {len(chunks)} 个分块)")
        print("="*80)
        
        for i, chunk in enumerate(chunks):
            print(f"\n📄 分块 #{i+1}/{len(chunks)}")
            print("-" * 60)
            
            # 获取来源信息
            source = chunk.metadata.get('source', '未知来源')
            chunk_id = chunk.metadata.get('chunk_id', f'chunk_{i}')
            file_type = chunk.metadata.get('file_type', '未知类型')
            
            # 打印元数据
            print(f"🆔 ID: {chunk_id}")
            print(f"📚 来源: {source} (类型: {file_type})")
            
            # 打印内容，保留原有格式
            content = chunk.page_content
            print("\n📝 内容:")
            print("-" * 40)
            # 使用print_wrapped方法，但保留段落结构
            if len(content) > 1000:
                # 只显示前1000字符，避免输出过长
                self.print_wrapped(content[:1000] + "...")
            else:
                self.print_wrapped(content)
            
            # 打印字符数和词数估计
            char_count = len(content)
            word_count = len(content.split())
            print(f"\n📊 统计: {char_count} 字符, 约 {word_count} 词")
            
            # 暂停以避免输出刷屏过快
            if i < len(chunks) - 1 and (i + 1) % 5 == 0:
                print("\n" + "-"*60)
                print(f"ℹ️  已显示 {i+1} 个分块，按回车继续查看下一个分块，或等待3秒自动继续...")
                try:
                    import select
                    import sys
                    print("⏳ 等待3秒后自动继续...", end="", flush=True)
                    for _ in range(30):
                        if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                            input()  # 消耗掉输入
                            break
                        print(".", end="", flush=True)
                    print("\n继续显示分块内容...")
                except:
                    # 在不支持select的系统上（如Windows），简单等待
                    time.sleep(1)
        
        print("\n" + "="*80)
        print(f"✅ 已完成所有 {len(chunks)} 个文档分块的展示")
        print("="*80)

    def create_embeddings(self):
        """创建中文嵌入模型"""
        print("🧠 初始化中文嵌入模型...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
            model_kwargs={'device': 'cpu'},  # 如果有GPU，可以改为 'cuda'
            encode_kwargs={'normalize_embeddings': True},
            cache_folder="./model_cache"
        )
        
        print("✅ 嵌入模型初始化完成")
    
    def create_or_load_vector_db(self):
        """创建或加载向量数据库"""
        if not self.embeddings:
            self.create_embeddings()
        
        # 检查索引文件是否存在
        index_exists = (
            os.path.exists(self.index_dir) and 
            os.path.exists(os.path.join(self.index_dir, "index.faiss")) and
            os.path.exists(os.path.join(self.index_dir, "index.pkl"))
        )
        
        if index_exists:
            print("🔍 加载已有的向量数据库...")
            start_time = time.time()
            
            self.vector_db = FAISS.load_local(
                self.index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            elapsed_time = time.time() - start_time
            print(f"✅ 向量数据库加载完成，耗时: {elapsed_time:.2f}秒")

            # 检查数据库中文档数量
            doc_count = len(self.vector_db.index_to_docstore_id)
            print(f"📊 向量数据库中包含 {doc_count} 个文本块")
            if doc_count == 0:
                print("⚠️  警告：向量数据库为空！可能需要重新创建索引。")

        else:
            print("🆕 未找到现有向量数据库，需要创建新的")
            documents = self.load_documents()
            
            if not documents:
                print("❌ 没有找到可处理的文档，请先添加文档到目录")
                sys.exit(1)
            
            chunks = self.split_documents(documents)
            
            if not chunks:
                print("❌ 没有生成有效的文本块")
                sys.exit(1)
            
            print(f"📊 将创建包含 {len(chunks)} 个文本块的向量数据库")
            if len(chunks) < 5:
                print("⚠️  警告：文档数量较少，可能影响检索效果")

            print("💾 创建新的向量数据库...")
            start_time = time.time()
            
            # 确保索引目录存在
            if not os.path.exists(self.index_dir):
                os.makedirs(self.index_dir)
            
            self.vector_db = FAISS.from_documents(chunks, self.embeddings)
            self.vector_db.save_local(self.index_dir)
            
            elapsed_time = time.time() - start_time
            print(f"✅ 向量数据库创建完成，耗时: {elapsed_time:.2f}秒")
        
        # 创建检索器，启用相似度得分阈值功能
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 4, "score_threshold": 0.3}
        )
    
    def setup_llm(self):
        """配置通义千问大模型"""
        print("🤖 初始化通义千问大模型...")
        
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("❌ 未设置 DASHSCOPE_API_KEY 环境变量，请在 .env 文件中配置")
        
        self.llm = ChatTongyi(
            model_name="qwen3-max",  # 使用 qwen3-max 模型，兼容性更好
            dashscope_api_key=api_key,
            temperature=0.7,
            max_tokens=1500,
            top_p=0.8,
            verbose=True
        )
        
        print("✅ 通义千问模型初始化完成")
    
    def format_docs(self, docs: List[Document]) -> str:
        """格式化检索到的文档"""
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', '未知来源')
            content = doc.page_content.strip()
            if content:
                formatted_docs.append(f"[参考文档 #{i} - {source}]\n{content}")
        
        return "\n\n".join(formatted_docs) if formatted_docs else "未找到相关参考文档"
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取会话历史"""
        if session_id not in self.chat_history:
            self.chat_history[session_id] = ChatMessageHistory()
        return self.chat_history[session_id]
    
    def _log_llm_input(self, inputs: dict):
        """记录输入给LLM的内容，用于调试"""
        print("\n" + "="*80)
        print("🔍 输入给LLM的详细内容 (调试信息)")
        print("="*80)
        print(f"📚 检索到的上下文:\n{inputs['context']}\n")
        print(f"💬 用户当前问题: {inputs['question']}\n")
        
        # 格式化聊天历史
        chat_history = inputs.get('chat_history', [])
        if chat_history:
            formatted_history = []
            for i, msg in enumerate(chat_history, 1):
                role = "👤 用户" if isinstance(msg, HumanMessage) else "🤖 助手"
                formatted_history.append(f"{role} (消息#{i}):\n{msg.content}")
            print("🗨️  对话历史 (最近的交互):")
            print("\n".join(formatted_history))
        else:
            print("🗨️  对话历史: 无")
        print("="*80 + "\n")
        return inputs  # 返回输入，不改变数据流
    
    def setup_rag_chain(self):
        """设置现代化 RAG 链 - 修复字典输入问题"""
        if not self.vector_db or not self.llm or not self.retriever:
            raise ValueError("❌ 向量数据库、LLM 或检索器未初始化")
        
        print("🔗 配置 RAG 推理链...")
        
        # 创建提示模板 - 中文优化
        system_prompt = """你是一位专业、准确的AI助手，名为通义助手。请根据以下提供的上下文信息和对话历史回答用户的问题。

你的回答需要遵循以下原则：
1. 📌 **事实准确**：仅使用提供的上下文信息回答问题，不要编造信息
2. 📝 **简洁专业**：使用简洁、专业的中文回答
3. ❓ **诚实坦率**：如果上下文信息不足以回答问题，请明确告知用户
4. 🎯 **针对性**：直接回答用户问题，不要离题
5. 📚 **引用来源**：如果可能，提及信息来源
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "相关上下文信息:\n{context}\n\n用户当前问题: {question}\n\n请提供专业、准确的回答：")
        ])
        
        # 修复：重新设计 RAG 链，确保输入是字典
        # 不再使用 RunnableWithMessageHistory，改为手动管理历史
        def format_chat_history(messages):
            """将消息列表转换为可读格式"""
            formatted_history = []
            for message in messages:
                role = "用户" if isinstance(message, HumanMessage) else "助手"
                formatted_history.append(f"{role}: {message.content}")
            return "\n".join(formatted_history)
        
        # 定义处理输入的函数 - 确保始终是字典
        def prepare_inputs(input_dict):
            if isinstance(input_dict, str):
                # 如果是字符串，转换为字典
                return {"question": input_dict, "chat_history": []}
            return input_dict
        
        # 创建基本RAG链
        self.rag_chain = (
            # 首先确保输入是字典
            RunnableLambda(prepare_inputs)
            # 然后添加上下文
            | {
                "context": lambda x: self.format_docs(self.retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", [])
            }
            | RunnableLambda(self._log_llm_input)
            # 应用提示模板
            | prompt
            # 生成回答
            | self.llm
            # 解析为字符串
            | StrOutputParser()
        )
        
        print("✅ RAG 推理链配置完成 (修复了字典输入问题)")
    
    def get_relevant_docs(self, query: str) -> List[Document]:
        """获取相关文档用于展示"""
        # 使用 similarity_search_with_score 方法获取文档和得分
        docs_and_scores = self.vector_db.similarity_search_with_score(query, k=4)
        filtered_docs = [(doc, score) for doc, score in docs_and_scores if score >= 0.3]
        
        # 打印得分信息
        print("\n📈 相似度得分详情:")
        print("-" * 40)
        for i, (doc, score) in enumerate(docs_and_scores, 1):
            status = "✅ 通过" if score >= 0.3 else "❌ 过滤"
            print(f"   文档 #{i} - 得分: {score:.4f} ({status})")
        
        # 返回过滤后的文档（不含得分）
        return [doc for doc, score in filtered_docs]
    
    def query(self, question: str) -> Dict[str, Any]:
        """执行查询并返回完整结果 - 修复字典输入问题"""
        start_time = time.time()
        
        try:
            # 获取相关文档（用于展示）
            relevant_docs = self.get_relevant_docs(question)
            
            # 获取当前会话历史
            chat_history = self.get_session_history(self.session_id).messages
            
            print("\n🔍 正在检索相关文档...")
            for i, doc in enumerate(relevant_docs[:3], 1):  # 只显示前3个文档预览
                source = doc.metadata.get('source', '未知来源')
                content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                print(f"   📄 [文档 #{i}] {source}: {content_preview}")
            if len(relevant_docs) > 3:
                print(f"   ➕ 还有 {len(relevant_docs)-3} 个相关文档...")
            
            # 调用RAG链 (会触发_log_llm_input日志)
            print("\n🤖 通义助手正在处理请求 (详细输入将显示在下方)...")

            # 关键修复：确保传入的是字典
            response = self.rag_chain.invoke({
                "question": question,
                "chat_history": chat_history
            })
            
            # 确保响应是字符串
            answer = str(response) if not isinstance(response, str) else response

            elapsed_time = time.time() - start_time
            
            # 更新聊天历史
            self.get_session_history(self.session_id).add_user_message(question)
            self.get_session_history(self.session_id).add_ai_message(answer)
            
            return {
                "question": question,
                "answer": answer,
                "relevant_docs": relevant_docs,
                "processing_time": elapsed_time,
                "session_id": self.session_id
            }
        
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_details = traceback.format_exc()
            print(f"❌ 查询处理错误详情:\n{error_details}")
            return {
                "question": question,
                "answer": f"处理问题时出错: {str(e)}\n详细错误: {error_details}",
                "relevant_docs": [],
                "processing_time": elapsed_time,
                "session_id": self.session_id,
                "error": str(e)
            }
    
    def print_wrapped(self, text: str, width: int = 80):
        """打印自动换行的文本"""
        if not isinstance(text, str):
            text = str(text)

        for line in text.split('\n'):
            print(textwrap.fill(line, width=width))
    
    def show_document_preview(self, documents: List[Document], sample_size: int = 3):
        """展示文档预览"""
        if not documents:
            return
        
        print("\n" + "="*60)
        print("📖 文档预览 (前3个文档片段)")
        print("="*60)
        
        for i, doc in enumerate(documents[:sample_size], 1):
            print(f"\n📄 文档 #{i}")
            print("-" * 40)
            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            self.print_wrapped(content)
            
            # 显示元数据
            metadata = {k: v for k, v in doc.metadata.items() if k not in ['chunk_id', 'chunk_index']}
            if metadata:
                print(f"\n🔍 元数据: {metadata}")
        
        print("\n" + "="*60)

    def clear_session(self):
        """清除当前会话历史"""
        self.chat_history[self.session_id] = ChatMessageHistory()
        print("🧹 对话历史已清除")

def setup_environment():
    """设置环境和依赖"""
    print("⚙️  检查环境配置...")
    
    # 检查必要的目录
    required_dirs = ["documents", "faiss_index", "model_cache"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ 创建目录: {dir_name}")
    
    # 检查环境变量
    required_vars = ["DASHSCOPE_API_KEY", "USER_AGENT"]
    for var in required_vars:
        if not os.getenv(var):
            print(f"⚠️  环境变量 {var} 未设置")
            if var == "DASHSCOPE_API_KEY":
                print("   💡 请在 .env 文件中设置您的阿里云DashScope API密钥")
    
    print("✅ 环境配置检查完成")

def main():
    """主函数"""
    print("="*60)
    print("🚀 通义RAG智能问答系统 (已修复字典输入问题)")
    print("="*60)
    
    # 设置环境
    setup_environment()
    
    # 初始化RAG系统
    rag_system = RAGSystem()
    
    # 创建或加载向量数据库
    rag_system.create_or_load_vector_db()
    
    # 设置LLM
    rag_system.setup_llm()
    
    # 设置RAG链 (使用修复版本)
    rag_system.setup_rag_chain()
    
    # 交互式问答
    print("\n" + "="*60)
    print("💬 系统已准备就绪！输入您的问题")
    print("팁 输入 'exit' 退出，输入 'clear' 清除对话历史")
    print("="*60)
    
    while True:
        try:
            question = input("\n👤 用户: ").strip()
            
            if question.lower() == 'exit':
                print("\n👋 感谢使用通义RAG系统，再见!")
                break
            
            if question.lower() == 'clear':
                rag_system.clear_session()
                continue
            
            if not question:
                continue
            
            print("\n🤖 通义助手正在思考...", end="", flush=True)
            
            # 获取结果
            result = rag_system.query(question)
            
            # 清除"正在思考"提示
            print("\r" + " " * 40 + "\r", end="")
            
            # 显示回答
            print("\n🤖 通义助手:")
            print("-" * 40)
            rag_system.print_wrapped(result["answer"])
            print(f"\n⏱️  耗时: {result['processing_time']:.2f}秒")
            
            # 显示参考文档
            if result["relevant_docs"]:
                print("\n" + "="*60)
                print("📚 参考的相关文档:")
                print("="*60)
                
                for i, doc in enumerate(result["relevant_docs"], 1):
                    print(f"\n📄 相关文档 #{i}")
                    print("-" * 40)
                    content = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                    rag_system.print_wrapped(content)
                    
                    source = doc.metadata.get('source', '未知来源')
                    print(f"\n📍 来源: {source}")
        
        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断，再见!")
            break
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"\n❌ 发生严重错误: {str(e)}")
            print(f"详细错误信息:\n{error_details}")
            print("💡 请检查您的配置或联系技术支持")

if __name__ == "__main__":
    main()