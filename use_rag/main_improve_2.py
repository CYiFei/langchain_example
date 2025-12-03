import os
import sys
import time
import textwrap
import json
import traceback
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Callable, Union, Tuple, Set
import importlib
import numpy as np
from collections import Counter
import re
import jieba  # 用于中文分词

# 加载环境变量
load_dotenv()

# 设置 USER_AGENT (解决警告)
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "rag-application/1.0")

# 新式 LangChain 导入
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredImageLoader,
    UnstructuredEmailLoader
)
from langchain_community.document_loaders import NotebookLoader
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
from unstructured.partition.auto import partition  # 用于自动检测文件类型

# 新增导入 - 混合检索所需
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("⚠️  未安装 rank_bm25，将跳过关键词检索功能。建议安装: pip install rank_bm25")
    BM25Okapi = None

try:
    import jieba
except ImportError:
    print("⚠️  未安装 jieba，将使用简单分词。建议安装: pip install jieba")
    jieba = None

# 元数据文件路径
METADATA_FILE = "document_metadata.json"

class HybridRetriever:
    """混合检索器：结合语义检索(FAISS)和关键词检索(BM25)"""
    
    def __init__(self, vector_retriever, documents: List[Document], bm25_k: int = 4, vector_k: int = 4):
        self.vector_retriever = vector_retriever  # FAISS 检索器
        self.bm25_k = bm25_k  # BM25 检索结果数量
        self.vector_k = vector_k  # 向量检索结果数量
        self.alpha = 0.5  # 混合权重：0=纯关键词，1=纯语义，0.5=平均混合
        
        # 预处理文档用于BM25
        self._prepare_bm25(documents)
    
    def _chinese_tokenize(self, text: str) -> List[str]:
        """中文分词处理"""
        if jieba:
            # 使用jieba进行中文分词
            return [word for word in jieba.cut(text) if len(word.strip()) > 1 and not word.strip().isdigit()]
        else:
            # 简单分词：按标点和空格分割，保留中文字符
            text = re.sub(r'[^\w\u4e00-\u9fa5]', ' ', text)  # 移除非中文、非字母数字字符
            words = text.split()
            # 过滤掉太短的词和纯数字
            return [word for word in words if len(word) > 1 and not word.isdigit()]
    
    def _prepare_bm25(self, documents: List[Document]):
        """准备BM25索引"""
        print("🔍 准备关键词检索索引(BM25)...")
        
        # 提取所有文档内容
        self.documents = documents
        self.corpus = [doc.page_content for doc in documents]
        
        # 中文分词处理
        print("✂️  对文档进行中文分词处理...")
        tokenized_corpus = []
        
        for i, text in enumerate(self.corpus):
            tokens = self._chinese_tokenize(text)
            tokenized_corpus.append(tokens)
            if (i + 1) % 100 == 0:
                print(f"   ✅ 已处理 {i+1}/{len(self.corpus)} 个文档")
        
        # 创建BM25模型
        self.bm25 = BM25Okapi(tokenized_corpus) if BM25Okapi else None
        print(f"✅ 关键词检索索引准备完成，共处理 {len(self.corpus)} 个文档")
    
    def _bm25_search(self, query: str, k: int = 4) -> List[Document]:
        """使用BM25进行关键词检索"""
        if not self.bm25:
            return []
        
        # 对查询进行中文分词
        tokenized_query = self._chinese_tokenize(query)
        
        if not tokenized_query:
            return []
        
        # 获取BM25分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取前k个最高分的文档索引
        top_k_idx = np.argsort(scores)[::-1][:k]
        
        # 返回对应的文档
        results = []
        for idx in top_k_idx:
            if scores[idx] > 0:  # 只返回有相关性的结果
                doc = self.documents[idx]
                doc.metadata["bm25_score"] = float(scores[idx])  # 转换为Python float
                results.append(doc)
        
        return results
    
    def _reciprocal_rank_fusion(self, results_list: List[List[Document]], k: int = 60) -> List[Document]:
        """使用倒数排名融合算法(RRF)合并多个检索结果"""
        fused_scores = {}
        
        # 为每组结果计算RRF分数
        for i, results in enumerate(results_list):
            for rank, doc in enumerate(results):
                doc_id = doc.metadata.get("chunk_id", f"doc_{id(doc)}")
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (rank + k)
        
        # 按分数排序
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 获取融合后的文档
        fused_documents = []
        seen_ids = set()
        
        for doc_id, score in sorted_docs:
            # 找到对应的文档
            for results in results_list:
                for doc in results:
                    current_id = doc.metadata.get("chunk_id", f"doc_{id(doc)}")
                    if current_id == doc_id and current_id not in seen_ids:
                        doc.metadata["fusion_score"] = score
                        fused_documents.append(doc)
                        seen_ids.add(doc_id)
                        break
                if doc_id in seen_ids:
                    break
        
        return fused_documents
    
    def _weighted_fusion(self, semantic_results: List[Document], keyword_results: List[Document], top_k: int = 4) -> List[Document]:
        """加权融合语义检索和关键词检索结果"""
        # 创建文档ID到结果的映射
        doc_map = {}
        
        # 处理语义检索结果
        for rank, doc in enumerate(semantic_results):
            doc_id = doc.metadata.get("chunk_id", f"doc_{id(doc)}")
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
                doc.metadata["semantic_rank"] = rank + 1
                doc.metadata["keyword_rank"] = 0  # 初始化
        
        # 处理关键词检索结果
        for rank, doc in enumerate(keyword_results):
            doc_id = doc.metadata.get("chunk_id", f"doc_{id(doc)}")
            if doc_id in doc_map:
                # 更新已存在的文档
                doc_map[doc_id].metadata["keyword_rank"] = rank + 1
            else:
                # 添加新的文档
                doc.metadata["semantic_rank"] = 0
                doc.metadata["keyword_rank"] = rank + 1
                doc_map[doc_id] = doc
        
        # 计算融合分数
        for doc_id, doc in doc_map.items():
            semantic_rank = doc.metadata.get("semantic_rank", 0)
            keyword_rank = doc.metadata.get("keyword_rank", 0)
            
            # 避免除零
            semantic_score = 1 / semantic_rank if semantic_rank > 0 else 0
            keyword_score = 1 / keyword_rank if keyword_rank > 0 else 0
            
            # 加权融合
            fused_score = self.alpha * semantic_score + (1 - self.alpha) * keyword_score
            doc.metadata["fused_score"] = fused_score
        
        # 按融合分数排序
        sorted_docs = sorted(
            doc_map.values(),
            key=lambda x: x.metadata.get("fused_score", 0),
            reverse=True
        )
        
        # 返回top_k结果
        return sorted_docs[:top_k]
    
    def get_relevant_documents(self, query: str, top_k: int = 4) -> List[Document]:
        """执行混合检索"""
        start_time = time.time()
        
        # 1. 语义检索 (FAISS)
        semantic_results = self.vector_retriever.invoke(query, k=self.vector_k)
        
        # 2. 关键词检索 (BM25)
        keyword_results = self._bm25_search(query, k=self.bm25_k) if self.bm25 else []
        
        # 3. 混合融合两种结果
        if keyword_results:
            # 使用加权融合
            fused_results = self._weighted_fusion(semantic_results, keyword_results, top_k)
            fusion_method = "加权融合"
        else:
            # 仅使用语义检索
            fused_results = semantic_results[:top_k]
            fusion_method = "仅语义检索(缺少BM25)"
        
        # 4. 添加检索方法信息到元数据
        for doc in fused_results:
            doc.metadata["retrieval_method"] = fusion_method
        
        elapsed_time = time.time() - start_time
        print(f"🔍 混合检索完成: 语义检索({len(semantic_results)}), 关键词检索({len(keyword_results)}), "
              f"融合结果({len(fused_results)}), 耗时: {elapsed_time:.2f}秒")
        
        return fused_results
    
    def set_alpha(self, alpha: float):
        """设置混合权重，0=纯关键词，1=纯语义，0.5=平均混合"""
        self.alpha = max(0.0, min(1.0, alpha))
        print(f"⚖️  混合检索权重已设置: alpha={self.alpha} (语义检索权重), 1-alpha={1-self.alpha} (关键词检索权重)")

class RAGSystem:
    """现代化的 RAG 系统，支持多文档类型、增量更新和混合检索"""
    
    def __init__(self, documents_dir: str = "documents", index_dir: str = "faiss_index"):
        self.documents_dir = documents_dir
        self.index_dir = index_dir
        self.embeddings = None
        self.vector_db = None
        self.llm = None
        self.rag_chain = None
        self.retriever = None
        self.hybrid_retriever = None  # 混合检索器
        self.all_documents = []  # 保存所有文档用于BM25
        self.chat_history = {}
        self.session_id = f"session_{int(time.time())}"
        self.processed_metadata = self.load_processed_metadata()
        
        # 配置国内镜像
        self._configure_mirrors()
        
        # 检查依赖
        self._check_dependencies()
    
    def _configure_mirrors(self):
        """配置国内镜像源加速下载"""
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("✅ 已配置 HuggingFace 国内镜像源")
    
    def _check_dependencies(self):
        """检查并提示安装必要的依赖"""
        missing_packages = []
        
        # 检查 unstructured 库
        try:
            import unstructured
        except ImportError:
            missing_packages.append("unstructured[all-docs]")
        
        # 检查 pytesseract (OCR)
        try:
            import pytesseract
        except ImportError:
            missing_packages.append("pytesseract")
        
        # 检查 pdf2image (PDF图像处理)
        try:
            from pdf2image import convert_from_path
        except ImportError:
            missing_packages.append("pdf2image")
        
        # 检查 pandoc (文档格式转换)
        try:
            import pypandoc
        except ImportError:
            missing_packages.append("pypandoc")
        
        # 检查 jieba (中文分词)
        try:
            import jieba
        except ImportError:
            missing_packages.append("jieba")
        
        # 检查 BM25
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            missing_packages.append("rank_bm25")
        
        # 如果有缺失的包，显示安装建议
        if missing_packages:
            print("\n" + "="*60)
            print("⚠️  检测到缺失的依赖包，请安装以获得完整功能")
            print(f"🔧 推荐安装命令: pip install {' '.join(missing_packages)}")
            print("✨ 完整依赖支持: PPTX, Excel, HTML, Markdown, EPUB, 图像OCR, 中文分词, 关键词检索等")
            print("="*60)
    
    def load_processed_metadata(self) -> Dict[str, dict]:
        """加载已处理文档的元数据"""
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  加载元数据文件失败: {str(e)}，将创建新的元数据记录")
        return {}
    
    def save_processed_metadata(self):
        """保存已处理文档的元数据"""
        try:
            with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.processed_metadata, f, ensure_ascii=False, indent=2)
            print(f"✅ 已保存文档元数据到 {METADATA_FILE}")
        except Exception as e:
            print(f"❌ 保存元数据失败: {str(e)}")
    
    def get_document_metadata(self, file_path: str) -> dict:
        """获取文档的元数据，用于检测变化"""
        if not os.path.exists(file_path):
            return {}
        
        stat = os.stat(file_path)
        return {
            "file_path": file_path,
            "file_size": stat.st_size,
            "last_modified": stat.st_mtime,
            "last_processed": time.time()
        }
    
    def show_supported_file_types(self):
        """显示支持的文件类型"""
        print("\n" + "="*80)
        print("📋 系统支持的文档类型")
        print("="*80)
        print(f"📁 文档目录: {self.documents_dir}")
        print("\n✅ 原生支持的文件类型:")
        print("   • PDF (.pdf) - 使用 PyPDFLoader")
        print("   • 文本文件 (.txt) - UTF-8编码")
        print("   • Word文档 (.docx) - 使用 docx2txt")
        print("   • CSV表格 (.csv) - 表格数据处理")
        print("   • JSON文件 (.json) - 结构化数据")
        print("   • Jupyter笔记本 (.ipynb) - 代码和文档")
        
        print("\n✅ 高级支持 (需要额外依赖):")
        print("   • PowerPoint演示文稿 (.pptx) - 需要 unstructured")
        print("   • Excel电子表格 (.xlsx, .xls) - 需要 unstructured")
        print("   • HTML网页 (.html, .htm) - 需要 unstructured")
        print("   • Markdown文档 (.md, .markdown) - 需要 unstructured")
        print("   • XML文件 (.xml) - 需要 unstructured")
        print("   • EPUB电子书 (.epub) - 需要 epub 库")
        print("   • 电子邮件 (.eml, .msg) - 需要 unstructured")
        print("   • 图像文件 (.jpg, .jpeg, .png, .bmp) - 需要 pytesseract (OCR)")
        
        print("\n✅ 增强检索功能:")
        print("   • 混合检索 - 结合语义检索(FAISS)和关键词检索(BM25)")
        print("   • 中文分词优化 - 使用 jieba 提升中文检索效果")
        
        print("\n🔄 自动检测 (尝试处理未知文件类型):")
        print("   • 任何包含文本的文件 - 需要 unstructured")
        
        print("\n💡 使用提示:")
        print("   1. 将文档放入 'documents' 目录")
        print("   2. 系统会自动识别并处理支持的文件类型")
        print("   3. 对于图像OCR，需要安装Tesseract: https://github.com/tesseract-ocr/tesseract")
        print("="*80)

    def create_loader_for_file(self, file_path: str, file_ext: str) -> Optional[Callable]:
        """根据文件类型创建适当的加载器"""
        file_name = os.path.basename(file_path)
        
        # PDF 文件
        if file_ext == '.pdf':
            return PyPDFLoader(file_path)
        
        # 文本文件
        elif file_ext in ['.txt', '.text']:
            return TextLoader(file_path, encoding='utf-8')
        
        # Word 文档
        elif file_ext == '.docx':
            return Docx2txtLoader(file_path)
        
        # CSV 文件
        elif file_ext == '.csv':
            return CSVLoader(file_path, encoding='utf-8')
        
        # JSON 文件
        elif file_ext == '.json':
            # 尝试使用智能JSON加载
            def load_json_file():
                try:
                    # 尝试加载为普通JSON
                    loader = JSONLoader(
                        file_path=file_path,
                        jq_schema='.',
                        text_content=False
                    )
                    return loader.load()
                except Exception as e:
                    print(f"⚠️  标准JSON加载失败，尝试回退方法: {str(e)}")
                    # 回退方法：将整个JSON作为文本读取
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return [Document(
                        page_content=f"JSON文件内容:\n{content}",
                        metadata={"source": file_name, "file_type": ".json"}
                    )]
            return load_json_file
        
        # Jupyter Notebook
        elif file_ext == '.ipynb':
            return NotebookLoader(file_path, include_outputs=True, max_output_length=2000)
        
        # 检查是否安装了 unstructured
        try:
            import unstructured
            
            # PowerPoint
            if file_ext == '.pptx':
                return UnstructuredPowerPointLoader(file_path)
            
            # Excel
            elif file_ext in ['.xlsx', '.xls']:
                return UnstructuredExcelLoader(file_path, mode="elements")
            
            # HTML
            elif file_ext in ['.html', '.htm']:
                return UnstructuredHTMLLoader(file_path)
            
            # Markdown
            elif file_ext in ['.md', '.markdown']:
                return UnstructuredMarkdownLoader(file_path)
            
            # XML
            elif file_ext == '.xml':
                return UnstructuredXMLLoader(file_path)
            
            # 电子邮件
            elif file_ext in ['.eml', '.msg']:
                return UnstructuredEmailLoader(file_path)
            
            # 图像文件 (OCR)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                print(f"🖼️  启用OCR处理图像: {file_name} (可能需要几秒钟)")
                return UnstructuredImageLoader(file_path, strategy="ocr_only")
                
        except ImportError:
            # unstructured 未安装
            if file_ext in ['.pptx', '.xlsx', '.xls', '.html', '.htm', '.md', '.markdown', 
                           '.xml', '.eml', '.msg', '.jpg', '.jpeg', '.png', '.bmp']:
                print(f"⚠️  跳过文件 {file_name}：需要安装 unstructured 库支持此类型")
                return None
        
        # 不支持的文件类型
        return None

    def load_documents(self, file_paths: List[str] = None) -> List[Document]:
        """
        加载文档，可选择性地只加载指定文件
        如果 file_paths 为 None，则加载所有文档
        """
        if not os.path.exists(self.documents_dir):
            os.makedirs(self.documents_dir)
            print(f"📁 创建文档目录: {self.documents_dir}")
            self.show_supported_file_types()
            print("\n💡 请将支持的文档文件放入此目录，然后重新运行程序")
            return []
        
        print(f"📚 开始加载文档，目录: {self.documents_dir}")
        if file_paths:
            print(f"🔍 仅加载指定文件: {', '.join(os.path.basename(p) for p in file_paths)}")
        else:
            self.show_supported_file_types()
        
        documents = []
        processed_files = 0
        skipped_files = 0
        unsupported_files = 0
        
        # 确定要处理的文件列表
        files_to_process = file_paths if file_paths else os.listdir(self.documents_dir)
        
        # 遍历文件
        for item in files_to_process:
            # 如果是完整路径，直接使用；否则构建路径
            filepath = item if os.path.isabs(item) or file_paths else os.path.join(self.documents_dir, item)
            filename = os.path.basename(filepath)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if not os.path.isfile(filepath):
                continue
            
            print(f"\n📄 处理文件: {filename} ({file_ext})")
            loader = self.create_loader_for_file(filepath, file_ext)
            
            if loader is None:
                print(f"❌ 不支持的文件类型或无法创建加载器: {filename}")
                unsupported_files += 1
                continue
            
            try:
                # 根据loader类型调用适当的方法
                if callable(loader) and not hasattr(loader, 'load'):
                    # 这是一个返回文档列表的函数
                    docs = loader()
                else:
                    # 这是一个标准的LangChain加载器
                    docs = loader.load()
                
                # 添加元数据
                for doc in docs:
                    doc.metadata.update({
                        "source": filename,
                        "file_path": filepath,
                        "file_type": file_ext,
                        "file_size": os.path.getsize(filepath),
                        "processed_time": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                documents.extend(docs)
                processed_files += 1
                print(f"✅ 成功处理: {filename} ({len(docs)} 个文档片段)")
            
            except Exception as e:
                print(f"❌ 处理文件 {filename} 时出错: {str(e)}")
                print(f"堆栈跟踪: {traceback.format_exc()}")
                skipped_files += 1
                
                # 尝试回退方法：如果标准加载失败，尝试用unstructured自动检测
                if file_ext not in ['.pdf', '.txt', '.docx']:  # 这些已经有专门的加载器
                    try:
                        print(f"🔄 尝试使用自动检测处理: {filename}")
                        from unstructured.partition.auto import partition
                        
                        elements = partition(filepath)
                        content = "\n\n".join([str(element) for element in elements])
                        
                        if content.strip():
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": filename,
                                    "file_path": filepath,
                                    "file_type": "auto-detected" + file_ext,
                                    "processed_time": time.strftime("%Y-%m-%d %H:%M:%S")
                                }
                            )
                            documents.append(doc)
                            processed_files += 1
                            skipped_files -= 1  # 撤销之前的跳过计数
                            print(f"✅ 回退方法成功: {filename}")
                    except Exception as fallback_e:
                        print(f"❌ 回退方法也失败: {str(fallback_e)}")
        
        # 统计总结
        print(f"\n{'='*60}")
        print("📊 文档加载统计")
        print(f"{'='*60}")
        print(f"✅ 成功处理: {processed_files} 个文件")
        print(f"❌ 跳过/失败: {skipped_files} 个文件")
        print(f"🚫 不支持类型: {unsupported_files} 个文件")
        print(f"📈 总共加载: {len(documents)} 个文档片段")
        print(f"{'='*60}")
        
        if processed_files == 0 and not file_paths:
            print("\n💡 没有成功加载文档，建议:")
            print("  1. 检查 'documents' 目录是否有文件")
            print("  2. 查看是否需要安装额外依赖")
            print("  3. 确保文件不是损坏的或受密码保护")
        
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
            chunk_id = f"chunk_{hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]}"
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["chunk_index"] = i
        
        print(f"✅ 文档分割完成: 生成了 {len(chunks)} 个文本块")
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
    
    def detect_changed_documents(self) -> Tuple[List[str], List[str]]:
        """
        检测新增或修改的文档
        返回: (新增文档路径列表, 修改过的文档路径列表)
        """
        new_docs = []
        modified_docs = []
        
        # 检查目录中所有文件
        for filename in os.listdir(self.documents_dir):
            filepath = os.path.join(self.documents_dir, filename)
            
            if not os.path.isfile(filepath):
                continue
            
            # 获取当前文件元数据
            current_metadata = self.get_document_metadata(filepath)
            
            # 检查文件是否已处理过
            doc_key = hashlib.md5(filepath.encode()).hexdigest()
            if doc_key in self.processed_metadata:
                # 检查文件是否被修改
                processed_meta = self.processed_metadata[doc_key]
                if (processed_meta.get("file_size") != current_metadata["file_size"] or
                    abs(processed_meta.get("last_modified", 0) - current_metadata["last_modified"]) > 1):
                    modified_docs.append(filepath)
            else:
                # 新文件
                new_docs.append(filepath)
        
        return new_docs, modified_docs
    
    def update_vector_db(self, new_chunks: List[Document]):
        """将新文本块添加到向量数据库"""
        if not self.vector_db:
            print("❌ 向量数据库未初始化，无法更新")
            return
        
        if not new_chunks:
            print("ℹ️  没有新文本块需要添加到向量数据库")
            return
        
        print(f"🔄 更新向量数据库，添加 {len(new_chunks)} 个新文本块...")
        start_time = time.time()
        
        # 添加新文档到向量库
        self.vector_db.add_documents(new_chunks)
        
        # 保存更新后的索引
        self.vector_db.save_local(self.index_dir)
        
        elapsed_time = time.time() - start_time
        print(f"✅ 向量数据库更新完成，耗时: {elapsed_time:.2f}秒")
    
    def create_or_load_vector_db(self):
        """创建或加载向量数据库，支持增量更新"""
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
            
            # 检查是否有新文档或修改过的文档
            new_docs, modified_docs = self.detect_changed_documents()
            
            if new_docs or modified_docs:
                print("\n" + "="*60)
                print("🆕 检测到文档变更")
                print(f"  • 新增文档: {len(new_docs)} 个")
                print(f"  • 修改文档: {len(modified_docs)} 个")
                print("="*60)
                
                # 合并需要处理的文件
                files_to_update = new_docs + modified_docs
                update_confirmation = input("\n❓ 是否更新向量数据库以包含这些变更? (Y/n): ").strip().lower()
                
                if update_confirmation in ['', 'y', 'yes']:
                    # 加载变更的文档
                    updated_docs = self.load_documents(files_to_update)
                    
                    if updated_docs:
                        # 分割文档
                        new_chunks = self.split_documents(updated_docs)
                        
                        # 更新元数据
                        for filepath in files_to_update:
                            doc_key = hashlib.md5(filepath.encode()).hexdigest()
                            self.processed_metadata[doc_key] = self.get_document_metadata(filepath)
                        
                        # 更新向量数据库
                        self.update_vector_db(new_chunks)
                        
                        # 保存更新后的元数据
                        self.save_processed_metadata()
                        print("✅ 文档元数据已更新")
                    else:
                        print("❌ 未生成有效的文本块，跳过更新")
                else:
                    print("⏭️  跳过文档更新，使用现有向量数据库")
            else:
                print("✅ 未检测到文档变更，使用现有向量数据库")

        else:
            print("🆕 未找到现有向量数据库，需要创建新的")
            documents = self.load_documents()
            
            if not documents:
                print("❌ 没有找到可处理的文档，请先添加文档到目录")
                sys.exit(1)
            
            # 保存所有文档用于混合检索
            self.all_documents = documents
            
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
            
            # 保存文档元数据
            for filename in os.listdir(self.documents_dir):
                filepath = os.path.join(self.documents_dir, filename)
                if os.path.isfile(filepath):
                    doc_key = hashlib.md5(filepath.encode()).hexdigest()
                    self.processed_metadata[doc_key] = self.get_document_metadata(filepath)
            
            self.save_processed_metadata()
            
            elapsed_time = time.time() - start_time
            print(f"✅ 向量数据库创建完成，耗时: {elapsed_time:.2f}秒")
        
        # 创建标准检索器
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # 创建混合检索器 (需要所有文档)
        print("🔗 创建混合检索器 (语义检索 + 关键词检索)...")
        if not hasattr(self, 'all_documents') or not self.all_documents:
            # 如果没有保存所有文档，需要从向量库中获取
            self.all_documents = list(self.vector_db.docstore._dict.values())
        
        # 创建混合检索器
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.retriever,
            documents=self.all_documents,
            bm25_k=4,
            vector_k=4
        )
        print("✅ 混合检索器创建完成")
    
    def setup_llm(self):
        """配置通义千问大模型"""
        print("🤖 初始化通义千问大模型...")
        
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("❌ 未设置 DASHSCOPE_API_KEY 环境变量，请在 .env 文件中配置")
        
        self.llm = ChatTongyi(
            model_name="qwen3-max",  # 使用 qwen3-max 模型
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
        if not self.vector_db or not self.llm or not self.hybrid_retriever:
            raise ValueError("❌ 向量数据库、LLM 或混合检索器未初始化")
        
        print("🔗 配置 RAG 推理链（使用混合检索）...")
        
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
                "context": lambda x: self.format_docs(self.hybrid_retriever.get_relevant_documents(x["question"], top_k=4)),
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
        
        print("✅ RAG 推理链配置完成 (使用混合检索)")
    
    def get_relevant_docs(self, query: str) -> List[Document]:
        """获取相关文档用于展示（使用混合检索）"""
        if not self.hybrid_retriever:
            print("⚠️  混合检索器未初始化，使用标准检索器")
            return self.retriever.invoke(query)
        
        return self.hybrid_retriever.get_relevant_documents(query, top_k=4)
    
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
                retrieval_method = doc.metadata.get('retrieval_method', '未知方法')
                bm25_score = doc.metadata.get('bm25_score', 'N/A')
                semantic_rank = doc.metadata.get('semantic_rank', 'N/A')
                keyword_rank = doc.metadata.get('keyword_rank', 'N/A')
                
                content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                
                print(f"   📄 [文档 #{i}] {source} (方法: {retrieval_method})")
                if bm25_score != 'N/A':
                    print(f"      • BM25分数: {bm25_score:.4f}")
                if semantic_rank != 'N/A' and keyword_rank != 'N/A':
                    print(f"      • 排名: 语义#{semantic_rank}, 关键词#{keyword_rank}")
                print(f"      • 内容: {content_preview}")
            
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
    
    def adjust_retrieval_alpha(self, alpha: float):
        """调整混合检索的权重 alpha (0=纯关键词，1=纯语义)"""
        if hasattr(self, 'hybrid_retriever') and self.hybrid_retriever:
            self.hybrid_retriever.set_alpha(alpha)
            print(f"✅ 混合检索权重已调整: alpha={alpha:.2f}")
        else:
            print("⚠️  混合检索器未初始化，无法调整权重")

    def clear_session(self):
        """清除当前会话历史"""
        self.chat_history[self.session_id] = ChatMessageHistory()
        print("🧹 对话历史已清除")
    
    def rebuild_index(self):
        """重建整个向量索引"""
        print("\n" + "="*60)
        print("🔄 重建向量索引")
        print("="*60)
        
        confirm = input("⚠️  此操作将删除现有索引并重新创建。确定要继续吗? (Y/n): ").strip().lower()
        if confirm not in ['', 'y', 'yes']:
            print("⏭️  已取消重建操作")
            return
        
        # 备份现有索引（如果存在）
        if os.path.exists(self.index_dir) and os.listdir(self.index_dir):
            backup_dir = f"{self.index_dir}_backup_{int(time.time())}"
            os.rename(self.index_dir, backup_dir)
            print(f"✅ 已备份现有索引到: {backup_dir}")
        
        # 清除元数据以便重新处理所有文档
        self.processed_metadata = {}
        self.save_processed_metadata()
        
        # 重新创建索引
        self.vector_db = None
        self.create_or_load_vector_db()
        
        print("✅ 索引重建完成")
    
    def manual_update_index(self):
        """手动触发索引更新"""
        print("\n" + "="*60)
        print("🔄 手动更新向量索引")
        print("="*60)
        
        # 检测变更
        new_docs, modified_docs = self.detect_changed_documents()
        
        if not new_docs and not modified_docs:
            print("ℹ️  未检测到文档变更，无需更新")
            return
        
        print(f"  • 新增文档: {len(new_docs)} 个")
        print(f"  • 修改文档: {len(modified_docs)} 个")
        
        # 显示变更的文件
        print("\n📁 变更的文件:")
        for filepath in new_docs + modified_docs:
            print(f"  • {os.path.basename(filepath)}")
        
        confirm = input("\n❓ 是否更新向量数据库以包含这些变更? (Y/n): ").strip().lower()
        if confirm not in ['', 'y', 'yes']:
            print("⏭️  已取消更新操作")
            return
        
        # 加载变更的文档
        files_to_update = new_docs + modified_docs
        updated_docs = self.load_documents(files_to_update)
        
        if not updated_docs:
            print("❌ 未加载到任何文档，更新失败")
            return
        
        # 保存所有文档用于混合检索
        self.all_documents.extend(updated_docs)
        
        # 分割文档
        new_chunks = self.split_documents(updated_docs)
        
        if not new_chunks:
            print("❌ 未生成有效的文本块，更新失败")
            return
        
        # 更新元数据
        for filepath in files_to_update:
            doc_key = hashlib.md5(filepath.encode()).hexdigest()
            self.processed_metadata[doc_key] = self.get_document_metadata(filepath)
        
        # 更新向量数据库
        self.update_vector_db(new_chunks)
        
        # 保存更新后的元数据
        self.save_processed_metadata()
        
        # 重新创建检索器
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # 重新创建混合检索器
        print("🔗 重新创建混合检索器...")
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.retriever,
            documents=self.all_documents,
            bm25_k=4,
            vector_k=4
        )
        
        print("✅ 索引更新完成")

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
    print("🚀 通义RAG智能问答系统 (混合检索增强版)")
    print("✨ 新增功能: 结合语义检索(FAISS)和关键词检索(BM25)的混合检索")
    print("="*60)
    
    # 设置环境
    setup_environment()
    
    # 初始化RAG系统
    rag_system = RAGSystem()
    
    # 创建或加载向量数据库（自动处理增量更新）
    rag_system.create_or_load_vector_db()
    
    # 设置LLM
    rag_system.setup_llm()
    
    # 设置RAG链
    rag_system.setup_rag_chain()
    
    # 交互式问答
    print("\n" + "="*60)
    print("💬 系统已准备就绪！输入您的问题")
    print("팁 输入 'exit' 退出，输入 'clear' 清除对话历史")
    print("팁 输入 'types' 查看支持的文件类型")
    print("팁 输入 'update' 手动更新索引，输入 'rebuild' 重建整个索引")
    print("팁 输入 'alpha 0.7' 调整混合检索权重 (0.0=纯关键词, 1.0=纯语义, 默认0.5)")
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
            
            if question.lower() == 'types':
                rag_system.show_supported_file_types()
                continue
            
            if question.lower() == 'update':
                rag_system.manual_update_index()
                continue
            
            if question.lower() == 'rebuild':
                rag_system.rebuild_index()
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