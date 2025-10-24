# 🔗 Chain-of-Retrieval: Aspect-Guided Multi-Agent Retrieval Framework

> *Multi-aspect-guided iterative retrieval framework using full context of scientific papers*

<p align="center">
  <img src="" width="650">
</p>

---

## 📘 Overview

**Chain-of-Retrieval (CoR)** is a multi-agent retrieval framework that decomposes a full scientific paper into multiple research aspects — such as *methodology, research questions(motivations), and experimental results* — and performs retrieval via a **chain-of-query** process, while the retrieved results are then hierarchically aggregated to form robust interpretation of dynamic relations between pool of related research by taking the decaying semantic relations with retrieval depth increase.

🔗 **arXiv:** [arxiv.org/abs/2507.10057](https://arxiv.org/abs/2507.10057)  
📊 **Benchmark:** [SciFullBench/PatentFullBench](https://huggingface.co/datasets/Jackson0018/Paper2PaperRetrievalBench)

---

## 🚀 Setup & Usage

### 🧩 Environment Setup
```bash
git clone https://github.com/psw0021/Chain-of-Retrieval.git
cd Chain-of-Retrieval
conda env create -f environment.yml -n paper_retrieval
conda activate paper_retrieval
```

### Download Benchmark
```bash
python download_benchmark.py
unzip Paper2PaperRetrieval.zip -d .
```

### Download our DPO-trained Query Optimizers
** Llama-3.2-3B-Instruct + Jina-Embeddings-v2-Base-EN [Llama-3.2-3B-Instruct + Jina-Embeddings-v2-Base-EN ]() \\
** Llama-3.2-3B-Instruct + BGE-M3 [Llama-3.2-3B-Instruct + BGE-M3](https://huggingface.co/Jackson0018/Llama-3.2-3B-Instruct_BGE) \\
** Llama-3.2-3B-Instruct + Inf-Retriever-v1-1.5B ** [Llama-3.2-3B-Instruct + Inf-Retriever-v1-1.5B](https://huggingface.co/Jackson0018/Llama-3.2-3B-Instruct_INFV) \\
** QWEN-2.5-3B-Instruct + Jina-Embeddings-v2-Base-EN ** [QWEN-2.5-3B-Instruct + Jina-Embeddings-v2-Base-EN]() \\
** QWEN-2.5-3B-Instruct + BGE-M3 ** [QWEN-2.5-3B-Instruct + BGE-M3](https://huggingface.co/Jackson0018/Qwen2.5-3B-Instruct_BGE) \\
** QWEN-2.5-3B-Instruct + Inf-Retriever-v1-1.5B ** [QWEN-2.5-3B-Instruct + Inf-Retriever-v1-1.5B](https://huggingface.co/Jackson0018/Qwen2.5-3B-Instruct_INFV) \\

```bash
python download_query_optimizers.py
```

### Run Evaluation using Llama-based DPO-trained Query Optimizers (only for SciFullBench)
```bash
bash Scripts/deploy_vllm_method_agent.sh
bash Scripts/deploy_vllm_experiment_agent.sh
bash Scripts/deploy_vllm_research_question_agent.sh

bash Scripts/inference_QoA_parallel_ai.sh
```

### Run Evaluation using QWEN-based DPO-trained Query Optimizers (only for SciFullBench)
```bash
bash Scripts/deploy_vllm_method_agent_QWEN.sh
bash Scripts/deploy_vllm_experiment_agent_QWEN.sh
bash Scripts/deploy_vllm_research_question_agent_QWEN.sh

bash Scripts/inference_QoA_parallel_ai.sh
```

### Run Evaluation using untrained Query Optimizers on SciFullBench
```bash
bash Scripts/inference_QoA_parallel_ai.sh
```

### Run Evaluation using untrained Query Optimizers for PatentFullBench
```bash
bash Scripts/inference_QoA_parallel_patents.sh
```

### How to reproduce results using SciMult
```bash
bash Scripts/inference_QoA_parallel_ai_SciMult.sh
```

