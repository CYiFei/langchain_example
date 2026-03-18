import asyncio
import aiohttp
import time
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime
import statistics

@dataclass
class TestResult:
    request_id: int
    status_code: int
    response_time: float
    success: bool
    error_message: str = ""

class ConcurrentTestClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        
    async def send_classification_request(
        self, 
        session: aiohttp.ClientSession, 
        request_id: int,
        text: str,
        classification_type: str,
        categories: List[str]
    ) -> TestResult:
        """发送单个分类请求"""
        start_time = time.time()
        
        payload = {
            "text": text,
            "classification_type": classification_type,
            "categories": categories
        }
        
        try:
            async with session.post(
                f"{self.base_url}/classify",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result_data = await response.json()
                    return TestResult(
                        request_id=request_id,
                        status_code=response.status,
                        response_time=response_time,
                        success=True
                    )
                else:
                    error_text = await response.text()
                    return TestResult(
                        request_id=request_id,
                        status_code=response.status,
                        response_time=response_time,
                        success=False,
                        error_message=error_text
                    )
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return TestResult(
                request_id=request_id,
                status_code=0,
                response_time=response_time,
                success=False,
                error_message="Request timeout"
            )
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                request_id=request_id,
                status_code=0,
                response_time=response_time,
                success=False,
                error_message=str(e)
            )
    
    async def run_concurrent_test(
        self,
        concurrent_requests: int = 10,
        total_requests: int = 100,
        test_data: List[Dict] = None
    ):
        """运行并发测试"""
        if test_data is None:
            test_data = [
                {
                    "text": "这款手机续航特别好，用了一天还有 50% 的电，就是系统偶尔有点卡",
                    "classification_type": "情感分类",
                    "categories": ["正面", "负面", "中性"]
                },
                {
                    "text": "我的订单已经付款 3 天了还没发货，能帮我查一下原因吗？",
                    "classification_type": "用户意图分类",
                    "categories": ["咨询产品", "投诉问题", "寻求帮助", "建议反馈"]
                },
                {
                    "text": "产品质量非常差，用了两天就坏了，客服也不理人",
                    "classification_type": "情感分类",
                    "categories": ["正面", "负面", "中性"]
                },
                {
                    "text": "希望能增加更多的颜色选择，现在选项太少了",
                    "classification_type": "用户意图分类",
                    "categories": ["咨询产品", "投诉问题", "寻求帮助", "建议反馈"]
                }
            ]
        
        print(f"\n{'='*60}")
        print(f"并发测试开始")
        print(f"{'='*60}")
        print(f"并发数：{concurrent_requests}")
        print(f"总请求数：{total_requests}")
        print(f"服务端地址：{self.base_url}")
        print(f"{'='*60}\n")
        
        self.results = []
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for i in range(total_requests):
                test_case = test_data[i % len(test_data)]
                task = self.send_classification_request(
                    session,
                    request_id=i + 1,
                    **test_case
                )
                tasks.append(task)
            
            semaphore = asyncio.Semaphore(concurrent_requests)
            
            async def limited_task(task):
                async with semaphore:
                    return await task
            
            results = await asyncio.gather(*[limited_task(task) for task in tasks])
            self.results.extend(results)
        
        total_time = time.time() - start_time
        
        self.print_statistics(total_time)
        
        return self.results
    
    def print_statistics(self, total_time: float):
        """打印统计信息"""
        if not self.results:
            print("没有测试结果")
            return
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        response_times = [r.response_time for r in self.results]
        successful_times = [r.response_time for r in successful]
        
        print(f"\n{'='*60}")
        print(f"测试结果统计")
        print(f"{'='*60}")
        print(f"总耗时：{total_time:.2f} 秒")
        print(f"总请求数：{len(self.results)}")
        print(f"成功请求：{len(successful)} ({len(successful)/len(self.results)*100:.2f}%)")
        print(f"失败请求：{len(failed)} ({len(failed)/len(self.results)*100:.2f}%)")
        print(f"QPS (Queries Per Second): {len(self.results)/total_time:.2f}")
        print(f"\n响应时间统计 (秒):")
        print(f"  最小值：{min(response_times):.3f}")
        print(f"  最大值：{max(response_times):.3f}")
        print(f"  平均值：{statistics.mean(response_times):.3f}")
        print(f"  中位数：{statistics.median(response_times):.3f}")
        print(f"  90% 分位：{sorted(response_times)[int(len(response_times)*0.9)]:.3f}")
        print(f"  95% 分位：{sorted(response_times)[int(len(response_times)*0.95)]:.3f}")
        print(f"  99% 分位：{sorted(response_times)[int(len(response_times)*0.99)]:.3f}")
        
        if successful_times:
            print(f"\n成功请求响应时间统计 (秒):")
            print(f"  平均值：{statistics.mean(successful_times):.3f}")
            print(f"  中位数：{statistics.median(successful_times):.3f}")
        
        if failed:
            print(f"\n失败请求详情:")
            for result in failed[:10]:
                print(f"  请求 {result.request_id}: {result.error_message}")
        
        print(f"\n{'='*60}\n")
    
    async def health_check(self):
        """健康检查"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✓ 服务健康检查通过：{data}")
                        return True
                    else:
                        print(f"✗ 服务健康检查失败：HTTP {response.status}")
                        return False
        except Exception as e:
            print(f"✗ 无法连接到服务：{e}")
            return False


async def main():
    client = ConcurrentTestClient("http://localhost:8000")
    
    print(f"\n测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    health_ok = await client.health_check()
    if not health_ok:
        print("\n请先启动服务端！")
        return
    
    await asyncio.sleep(1)
    
    await client.run_concurrent_test(
        concurrent_requests=10,
        total_requests=100
    )
    
    await asyncio.sleep(2)
    
    print("\n开始压力测试...")
    await client.run_concurrent_test(
        concurrent_requests=50,
        total_requests=500
    )


if __name__ == "__main__":
    asyncio.run(main())