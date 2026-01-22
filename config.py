"""
统一路径配置文件
所有路径都应从此文件导入，以便于管理和维护
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型路径
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Vicuna模型路径
VICUNA_MODEL_PATH = os.path.join(MODELS_DIR, "vicuna-7b-v1.3")

# Llama2模型路径
LLAMA2_MODEL_PATH = os.path.join(MODELS_DIR, "llama2-7b-chat")

# Llama3模型路径
LLAMA3_MODEL_PATH = os.path.join(MODELS_DIR, "llama3-8b")

# Hagrid相关路径
HAGRID_QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "hagrid_questions.jsonl")
LOG_FILE_DIR = os.path.join(PROJECT_ROOT, "log_file")

# 其他基准测试路径
LAWBENCH_CORPUS_PATH = os.path.join(PROJECT_ROOT, "lawbench", "corpus.json")

# 其他可能的路径
WIKIPEDIA_DPR_IVFPQ_INDEX_PATH = os.path.join(PROJECT_ROOT, "embedding", "wikipedia_dpr_ivfpq.index")

# 特定数据文件路径
WIKIPEDIA_DPR_SPLITTED_PKL_PATH = os.path.join(PROJECT_ROOT, "wikipedia_dpr_splitted.pkl")