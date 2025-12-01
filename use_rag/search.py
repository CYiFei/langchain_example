# find_retrieval_qa.py
import importlib
import pkgutil
import sys
import inspect

def find_class_in_package(package_name, class_name):
    """在指定包及其子模块中查找类"""
    try:
        package = importlib.import_module(package_name)
        print(f"成功导入包: {package_name}")
        
        # 检查包本身
        if hasattr(package, class_name):
            print(f"✓ 在 {package_name} 中找到 {class_name}")
            return getattr(package, class_name)
        
        # 递归检查所有子模块
        print(f"搜索 {package_name} 的子模块...")
        for _, module_name, _ in pkgutil.walk_packages(package.__path__, package_name + '.'):
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    print(f"✓ 在 {module_name} 中找到 {class_name}")
                    return getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                continue
                
    except ImportError as e:
        print(f"无法导入 {package_name}: {str(e)}")
    
    print(f"✗ 未在 {package_name} 及其子模块中找到 {class_name}")
    return None

# 搜索可能的包
search_packages = [
    "langchain",
    "langchain_community",
    "langchain.chains",
    "langchain_community.chains",
    "langchain_community.chains.retrieval_qa",
    "langchain.chains.retrieval_qa"
]

print("Python 版本:", sys.version)
print("-" * 50)

RetrievalQA = None
for package in search_packages:
    result = find_class_in_package(package, "RetrievalQA")
    if result:
        RetrievalQA = result
        break

if not RetrievalQA:
    print("\n尝试搜索相似类名...")
    # 尝试搜索其他可能的类名
    possible_classes = ["RetrievalQAChain", "QAWithSourcesChain", "RetrievalQA"]
    for package in ["langchain.chains", "langchain_community.chains"]:
        try:
            module = importlib.import_module(package)
            for name, obj in inspect.getmembers(module):
                if any(cls in name for cls in possible_classes) and inspect.isclass(obj):
                    print(f"找到相似类: {name} in {package}")
        except ImportError:
            continue

print("-" * 50)