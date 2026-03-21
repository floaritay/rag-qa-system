
"""
RAG系统评估脚本
用于计算Top-N召回率、精确率等指标
"""
import sys
import os
import json
from typing import List, Dict, Set
import time

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入后端模块
from backend.main import (
    init_vectorstore,
    retriever as main_retriever
)

class RAGEvaluator:
    def __init__(self):
        print("初始化评估器...")
        success = init_vectorstore()
        if not success:
            raise Exception("知识库初始化失败，请确保course_knowledge_base目录存在且有效")
        
        self.retriever = main_retriever
        print("评估器初始化完成")
    
    def evaluate_precision_at_k(self, query: str, relevant_keywords: List[str], k: int = 3) -> float:
        """
        计算Top-K精确率（Precision@K）
        
        精确率 = 检索到的相关文档数 / 检索到的文档总数
        
        参数:
            query: 查询问题
            relevant_keywords: 相关关键词列表
            k: 检索的Top-K数量
            
        返回:
            precision: 精确率 (0-1之间)
        """
        try:
            docs = self.retriever.invoke(query)[:k]
            
            if not docs:
                return 0.0
            
            relevant_count = 0
            for doc in docs:
                content = doc.page_content.lower()
                is_relevant = any(keyword.lower() in content for keyword in relevant_keywords) # 每个文档只要匹配至少一个关键词即视为相关
                if is_relevant:
                    relevant_count += 1
            
            precision = relevant_count / len(docs)
            return precision
            
        except Exception as e:
            print(f"计算精确率时出错: {e}")
            return 0.0
    
    def evaluate_recall_at_k(self, query: str, relevant_keywords: List[str], k: int = 3) -> float:
        """
        计算Top-K召回率（Recall@K）
        
        召回率 = 在检索结果中找到的不同关键词数 / 总关键词数
        
        参数:
            query: 查询问题
            relevant_keywords: 相关关键词列表（用于判断检索结果是否相关）
            k: 检索的Top-K数量
            
        返回:
            recall: 召回率 (0-1之间)
        """
        try:
            docs = self.retriever.invoke(query)[:k]
            
            if not docs or not relevant_keywords:
                return 0.0
            
            found_keywords: Set[str] = set()
            all_content = ' '.join([doc.page_content.lower() for doc in docs])
            
            for keyword in relevant_keywords:
                if keyword.lower() in all_content:
                    found_keywords.add(keyword.lower())
            
            recall = len(found_keywords) / len(relevant_keywords)
            return recall
            
        except Exception as e:
            print(f"计算召回率时出错: {e}")
            return 0.0
    
    def evaluate_batch(self, test_cases: List[Dict]) -> Dict:
        """
        批量评估测试用例
        
        参数:
            test_cases: 测试用例列表，每个元素包含query和relevant_keywords
            
        返回:
            results: 包含各项指标的统计结果
        """
        results = {
            'precision@1': [],
            'precision@3': [],
            'precision@5': [],
            'recall@1': [],
            'recall@3': [],
            'recall@5': [],
            'avg_response_time': []
        }
        
        for i, case in enumerate(test_cases):
            print(f"\n处理测试用例 {i+1}/{len(test_cases)}...")
            query = case['query']
            relevant_keywords = case['relevant_keywords']
            
            start_time = time.time()
            
            for k in [1, 3, 5]:
                precision = self.evaluate_precision_at_k(query, relevant_keywords, k)
                recall = self.evaluate_recall_at_k(query, relevant_keywords, k)
                results[f'precision@{k}'].append(precision)
                results[f'recall@{k}'].append(recall)
            
            end_time = time.time()
            results['avg_response_time'].append(end_time - start_time)
            
            print(f"  查询: {query[:50]}...")
            print(f"  Precision@3: {results['precision@3'][-1]:.2f}")
            print(f"  Recall@3: {results['recall@3'][-1]:.2f}")
        
        # 计算平均值
        summary = {}
        for metric, values in results.items():
            if values:
                summary[metric] = sum(values) / len(values)
        
        return summary
    
    def print_report(self, test_cases: List[Dict]) -> str:
        """
        生成评估报告
        """
        print("\n" + "="*60)
        print("RAG系统评估报告")
        print("="*60)
        
        results = self.evaluate_batch(test_cases)
        
        print("\n平均指标:")
        print("-"*60)
        for metric, value in results.items():
            if metric == 'avg_response_time':
                print(f"{metric:<25}: {value:.4f} 秒")
            else:
                print(f"{metric:<25}: {value:.4f} ({value*100:.2f}%)")
        
        print("\n" + "="*60)
        return json.dumps(results, indent=2, ensure_ascii=False)

# 示例测试用例
def get_sample_test_cases():
    """
    生成示例测试用例
    请根据您的实际课程资料修改这些测试用例
    """
    return [
        {
            "query": "什么是矩阵制造车间？",
            "relevant_keywords": ["矩阵", "制造车间"]
        },
        {
            "query": "多AGV调度系统研究的核心算法是什么？",
            "relevant_keywords": ["nsga", "遗传算法", "多目标", "优化", "调度"]
        },
        {
            "query": "单载量无人搬运车优化调度方法是怎样的？",
            "relevant_keywords": ["单载量", "无人搬运车", "优化调度", "步骤", "结果", "结论"]
        }
    ]

if __name__ == "__main__":
    print("启动RAG系统评估...")
    
    # 设置环境变量（如果需要）
    if not os.getenv("BAILIAN_API_KEY"):
        print("警告: 请先设置BAILIAN_API_KEY环境变量")
        print("使用方式: set BAILIAN_API_KEY=你的API密钥")
        sys.exit(1)
    
    try:
        evaluator = RAGEvaluator()
        
        # 使用示例测试用例进行评估
        test_cases = get_sample_test_cases()
        print(f"\n加载了 {len(test_cases)} 个测试用例")
        
        # 运行评估
        evaluator.print_report(test_cases)
        
        print("\n提示: 请根据您的实际课程资料修改 get_sample_test_cases() 函数中的测试用例")
        print("以获得更准确的评估结果")
        
    except Exception as e:
        print(f"\n评估失败: {e}")
        import traceback
        traceback.print_exc()
