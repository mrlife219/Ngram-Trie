import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from vector_store_util import *
import time
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import csv



splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)  


docs = []
docs_dict = []
cnt = 0
avg_len = 0
with open('your_wikipedia_file.csv', 'r') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    for row in tqdm(tsvreader):
        if cnt == 0:
            cnt += 1
            continue
        docs.append(row[1])
        docs_dict.append({'title': row[2], 'text': row[1]})
        avg_len += len(row[1])

print(len(docs))
print(avg_len/len(docs))
pickle.dump(docs_dict, open('wikipedia_dpr_splitted.pkl', 'wb'))
dataset_dir = 'wikipedia_dpr_splitted.pkl'
docs = pickle.load(open(dataset_dir, 'rb'))
start_time = time.time()
manager = VecRetrieval_IVFPQ(load_result=docs, dim=1024, top_k=5, embedding_store='./embedding/wikipedia_dpr_ivfpq.index', model_path='BAAI/bge-m3')

# manager.vec_storage()   
print("--- %s seconds ---" % (time.time() - start_time))
#     # 添加数据到索引（假设有一个包含向量的pickle文件）
# docs = pickle.load(open(dataset_dir, 'rb'))

# 执行搜索
start_time = time.time()
query = "What is the capital of Germany?"
top_k = 5
# texts = manager.search(query, top_k=top_k, docs=docs)
texts = manager.vec_retrieval_topk(query)
print('search_time:', time.time() - start_time)

for text in texts:
    print(text)

