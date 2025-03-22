# GraphRAG 

将知识图生成和查询聚焦总结相结合，以解决向量 RAG 在全局问题上的不足，支持对整个文本语料库的人类感知。研究显示，GraphRAG 在回答全面性和多样性上优于向量 RAG 基线，为处理大型文档集合的问答任务提供了有效方案。

## 步骤

### 安装必要的库
```bash
# RAG需要
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openai
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple langchain
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple langchain-community
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple faiss-cpu
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple langchain_openai
# GraphRAG需要
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple langchain-graphrag
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyvis
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple langchain-chroma
```
