# 核心思路：检索阶段单独评，生成阶段单独评，最后再看端到端。

# 测试规模和类型：
"""
测试集规模：
每 100 份文档至少对应 5-10 条测试问题

测试集类型（三类）:

1.确定性答案的
事实提取 - "A 款保险的免赔额是多少？" - 精确定位
多文档综合 - "比较 A 款和 B 款保险的保障范围" - 跨文档库表
推理判断 - "C 这个理赔案例符合哪款保险的赔付条件？" - 逻辑推理
时间过滤 - "公司最新的车险理赔流程有哪些变化？" - 时效推理

2.模棱两可的
模棱两可的提问 - "查下销售额是多少？" - 澄清依赖的上下文

3.拒答的
拒答问题 - "帮我算一下 D 这笔理赔金额" - "我无法回答"的问题很重要
"""

# 每条问题必须标注三样东西
## 标准答案（ground_truth_answer）：用来评估生成质量。
## 标准证据（ground_truth_chunks）：用来评估检索质量。
## 元信息：问题类型、难度等级。
{
     "question": "A款重疾险的等待期是多少天？",
     "ground_truth_answer": "A款重疾险的等待期为90天",
     "ground_truth_chunks": [
         {"doc_id": "policy_A_v3.pdf", "chunk_id": "chunk_47",
          "content": "本合同的等待期为90日，自合同生效之日起计算..."}
     ],
     "question_type": "factual",
     "difficulty": "easy"
 }

# 让 LLM 从文档中自动生成问答对（阅读文档片段，生成 QA），大约覆盖 70% 测试集，后续人工补充 30%

def generate_test_set_from_docs(docs: list[dict], llm: str = "gpt-4") -> list[dict]:
     """
     从文档自动生成测试集初稿。
     为什么不直接用 LLM 生成的结果？
     因为 LLM 生成的问题有两个问题：
     1. 倾向于生成"阅读理解"式的简单问题
     2. 可能生成文档中不存在的"幻觉问答对"
     必须人工审核后才能用。
     """
     test_items = []
     for doc in docs:
         prompt = f"""
         阅读以下文档片段，生成 3 个不同类型的问题和对应答案：
         1. 一个事实提取题（答案直接出现在文中）
         2. 一个推理题（需要结合文中多条信息推理）
         3. 一个该文档无法回答的问题（标注答案为"无法回答"）

         文档片段：
         {doc['content'][:2000]}

         输出格式：JSON数组
         """
         result = call_llm(llm, prompt)
         test_items.extend(parse_qa_pairs(result, doc))
     return test_items

# 检索阶段：
## 指标一：Recall@K（召回率）

def recall_at_k(retrieved_chunks: list[str], ground_truth_chunks: list[str],
                 k: int = 5) -> float:
     """
     Recall@K：相关文档有没有被找回来。
     为什么用 K=5？因为我们的系统给 LLM 的上下文默认传 Top-5。
     如果 Top-5 里没有正确文档，后面生成阶段再好也答不对。
     """
     top_k = set(retrieved_chunks[:k])
     relevant = set(ground_truth_chunks)
     if not relevant:
         return 1.0  # 没有标注证据的问题跳过
     return len(top_k & relevant) / len(relevant)

## 指标二：Precision@K（准确率）

def precision_at_k(retrieved_chunks: list[str], ground_truth_chunks: list[str],
                    k: int = 5) -> float:
     """
     Precision@K：找回来的东西有没有用。
     Recall 高但 Precision 低意味着"找到了但混了很多垃圾"。
     垃圾文档多了会干扰 LLM 生成，甚至引入幻觉。
     """
     top_k = set(retrieved_chunks[:k])
     relevant = set(ground_truth_chunks)
     if not top_k:
         return 0.0
     return len(top_k & relevant) / len(top_k)  # Recall和Precision经常是矛盾，召回更多文档可以提高Recall但也会引入不相关的内容拉低Precision

## 指标三：MRR（第一个相关文档排在第几位）

def mrr(retrieved_chunks: list[str], ground_truth_chunks: list[str]) -> float:
     """
     MRR 关注的是"最相关的文档有没有排在前面"。
     MRR=1 表示第一个结果就是对的，MRR=0.5 表示第二个才是。
     对用户体验影响很大——如果正确文档排在第1位 vs 第5位，
     LLM 的注意力分配是不一样的。
     """
     relevant = set(ground_truth_chunks)
     for i, chunk in enumerate(retrieved_chunks):
         if chunk in relevant:
             return 1.0 / (i + 1)
     return 0.0

# 生成阶段：
## 指标一：答案准确率（Correctness）不能只用精确匹配

def answer_correctness(prediction: str, ground_truth: str,
                        method: str = "llm_judge") -> float:
     """
     答案准确率评估。
     为什么不用 BLEU/ROUGE？
     因为 RAG 的答案往往是解释性文本，不是翻译，
     BLEU 这种 n-gram 匹配在这个场景下相关性很低。
     用 LLM-as-Judge 最靠谱。
     """
     if method == "f1":
         return compute_token_f1(prediction, ground_truth)
     elif method == "llm_judge":
         prompt = f"""
         判断以下回答是否正确回答了问题。

         标准答案：{ground_truth}
         模型回答：{prediction}

         评分标准（0-1）：
         - 1.0：完全正确，覆盖了标准答案的所有关键信息
         - 0.7：基本正确，但遗漏了部分关键信息
         - 0.3：部分正确，有关键错误或严重遗漏
         - 0.0：完全错误或答非所问
         """
         return float(call_llm("gpt-4", prompt))

## 指标二：忠实度（Faithfulness）即有出处，检测幻觉的核心指标。

def faithfulness(answer: str, retrieved_contexts: list[str]) -> float:
     """
     忠实度：答案是否忠实于检索到的文档。
     这是 RAG 系统最重要的生成指标——
     如果忠实度低，说明模型在"编"而不是在"引用"。
     """
     prompt = f"""
     将以下回答拆分为独立的事实声明，然后逐条检查每条声明
     是否能在提供的参考文档中找到支持。

     回答：{answer}

     参考文档：
     {chr(10).join(f'[{i+1}] {ctx}' for i, ctx in enumerate(retrieved_contexts))}

     输出格式：
     - 总声明数：N
     - 有支持的声明数：M
     - 忠实度分数：M/N
     - 无支持的声明列表：[...]
     """
     result = call_llm("gpt-4", prompt)
     return parse_faithfulness_score(result)

## 指标三：答案完整性（Completeness）

"""答案有没有覆盖标准答案中的所有关键信息点。"""

# 评估流程 全自动化

def run_full_evaluation(rag_system, test_set: list[dict]) -> dict:
     """
     完整的自动化评估流水线。
     输入：RAG 系统 + 测试集
     输出：分阶段的指标报告
     """
     retrieval_metrics = {"recall@5": [], "precision@5": [], "mrr": [], "redundancy": []}
     generation_metrics = {"correctness": [], "faithfulness": [], "completeness": []}

     for item in test_set:
         # 第一步：只跑检索，不跑生成
         retrieved = rag_system.retrieve(item["question"], top_k=5)
         retrieved_ids = [r["chunk_id"] for r in retrieved]
         gt_ids = [c["chunk_id"] for c in item["ground_truth_chunks"]]

         retrieval_metrics["recall@5"].append(recall_at_k(retrieved_ids, gt_ids, 5))
         retrieval_metrics["precision@5"].append(precision_at_k(retrieved_ids, gt_ids, 5))
         retrieval_metrics["mrr"].append(mrr(retrieved_ids, gt_ids))
         retrieval_metrics["redundancy"].append(redundancy_rate(retrieved))

         # 第二步：跑完整流程，评估生成
         answer = rag_system.generate(item["question"])
         contexts = [r["content"] for r in retrieved]

         generation_metrics["correctness"].append(
             answer_correctness(answer, item["ground_truth_answer"])
         )
         generation_metrics["faithfulness"].append(
             faithfulness(answer, contexts)
         )
         generation_metrics["completeness"].append(
             answer_completeness(answer, item["ground_truth_answer"])
         )

     # 汇总
     report = {}
     for stage, metrics in [("retrieval", retrieval_metrics),
                            ("generation", generation_metrics)]:
         report[stage] = {
             name: round(sum(values) / len(values), 4)
             for name, values in metrics.items()
         }
     return report

## 输出示例：
====== RAG 系统评估报告 ======
 测试集：200 条 | 日期：2026-05-14

 【检索阶段】
   Recall@5:     0.8920  (↑ 0.03 vs 上次)
   Precision@5:  0.7640
   MRR:          0.8150
   冗余率:        0.1230  (↓ 0.05 vs 上次)

 【生成阶段】
   答案准确率:    0.8450
   忠实度:        0.9180
   完整性:        0.7820

 【端到端】
   综合准确率:    0.8250

 【问题类型分析】
   事实提取:  准确率 0.93 | 忠实度 0.95
   多文档综合: 准确率 0.78 | 忠实度 0.88
   推理判断:  准确率 0.72 | 忠实度 0.90
   时效性:   准确率 0.81 | 忠实度 0.92
   拒答识别:  准确率 0.65 | 忠实度 0.98
 ===============================
### 分维度洞察："拒答识别"准确率只有 65%；"多文档综合"准确率 78%，跨文档整合能力需要加强

# 评估框架（想开箱即用）

## RAGAS，内置了 Faithfulness、Answer Relevancy、Context Precision、Context Recall 四维度自动评估，适合快速迭代，后转自建测试集。

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
  
 # 准备评估数据
 eval_data = {
     "question": questions,
     "answer": predictions,
     "contexts": retrieved_contexts,
     "ground_truth": ground_truths
 }

 # 一行代码跑评估
 result = evaluate(eval_data,
                   metrics=[faithfulness, answer_relevancy,
                            context_precision, context_recall])
 print(result)

