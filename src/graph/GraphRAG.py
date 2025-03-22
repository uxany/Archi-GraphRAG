import pickle
import pandas as pd
from langchain_graphrag.indexing import TextUnitExtractor, IndexerArtifacts
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
import keys
from langchain_core.documents import Document
from langchain_graphrag.indexing.graph_generation import EntityRelationshipExtractor, GraphsMerger, EntityRelationshipDescriptionSummarizer, GraphGenerator
from langchain_community.cache import SQLiteCache
from pyvis.network import Network
from langchain_graphrag.indexing.graph_clustering.leiden_community_detector import HierarchicalLeidenCommunityDetector
from langchain_graphrag.indexing.artifacts_generation import (
    CommunitiesReportsArtifactsGenerator,
    EntitiesArtifactsGenerator,
    RelationshipsArtifactsGenerator,
    TextUnitsArtifactsGenerator,
)
from langchain_graphrag.indexing.report_generation import (
    CommunityReportGenerator,
    CommunityReportWriter,
)
from langchain_graphrag.indexing import SimpleIndexer, TextUnitExtractor
from langchain_chroma.vectorstores import Chroma as ChromaVectorStore
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import networkx as nx
import re

import warnings

warnings.filterwarnings("ignore")



# 读取pdf到一个Document对象中
def pdf_to_doc(pdf_path):
    pdf = PdfReader(pdf_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

    doc = Document(page_content=text)
    return doc


doc1 = pdf_to_doc('./海龟交易法则_130-135.pdf')
doc2 = pdf_to_doc('./海龟交易法则_136-142.pdf')

# 递归地将文本分割成块
spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
# 将分割后的文本块封装为结构化数据（如DataFrame）
text_unit_extractor = TextUnitExtractor(text_splitter=spliter)
textunit_df = text_unit_extractor.run([doc1, doc2])


from langchain_text_splitters import RecursiveCharacterTextSplitter

# 初始化分割器
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],  # 分隔符列表
    chunk_size=10,  # 每块文本的最大长度
    chunk_overlap=2  # 块之间的重叠长度
)

# 示例文本
text = "这是一个示例文本\n\n它包含多个句子\n\n用于演示如何使用\n\nRecursiveCharacterTextSplitter\n\n进行文本分割。"

# 分割文本
text_splitter.split_text(text)