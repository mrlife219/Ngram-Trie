# coding:utf-8

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import time
import faiss
import numpy as np
import pandas as pd
from text2vec import SentenceModel
import bm25s
import pickle
from pathlib import Path

class FaissIndexManager:
    def __init__(self, embedding_dim, index_folder, model_name, index_path=None):
        """
        初始化FaissIndexManager。
        
        :param embedding_dim: 向量的维度
        :param index_folder: 存放索引文件的文件夹路径
        :param model_name: 选择的embedding模型名称
        """
        self.embedding_dim = embedding_dim
        self.index_folder = Path(index_folder)
        self.model_name = model_name
        self.model = SentenceModel(self.model_name)
        self.index_folder.mkdir(parents=True, exist_ok=True)
        self.index_file = []
        if self.index_folder.exists():
            for file in self.index_folder.glob('*.index'):
                self.index_file.append(file)
        if len(self.index_file) != 0:
            self.read_indexes()
        self.batch_size = 100000  # 每批次处理的数据量
        self.index_path = index_path

    def _create_index(self):
        """创建一个新的索引"""
        return faiss.IndexFlatL2(self.embedding_dim)
    
    def _random_gen(self, num_samples):
        """随机采样数据"""
        return np.random.rand(num_samples, self.embedding_dim).astype('float32')
    
    def load_index_from_pickle(self, index_path):
        """
        从pickle文件中读取索引。
        
        :param index_path: 索引文件的路径
        """
        if os.path.exists(index_path):
            with open(index_path, 'rb') as f:
                self.data = pickle.load(f) 

    def add_vectors_from_pickle(self, pickle_file_path, num_groups=7):
        """
        从pickle文件中读取向量并分组添加到多个索引中。
        
        :param pickle_file_path: 包含向量的pickle文件路径
        :param num_groups: 分成多少组进行存储，默认为4
        """
        if hasattr(self, 'data') is False: # 假设data是pickle形式
            self.load_index_from_pickle(pickle_file_path)

        total_vectors = len(self.data)
        vectors_per_group = total_vectors // num_groups
        
        for group_idx in range(num_groups):
            start = group_idx * vectors_per_group
            end = (group_idx + 1) * vectors_per_group if group_idx < num_groups - 1 else total_vectors
            group_data = self.data[start:end]
            group_text = [i['text'] for i in group_data]
            group_embedding = self.model.encode(group_text)
            
            index = self._create_index()
            self._add_vectors_in_batches(index, group_embedding)
            index_path = self.index_folder / f'index_wikipedia_group_{group_idx}.index'
            faiss.write_index(index, str(index_path))
            print(f"Index group {group_idx} created and saved to {index_path}")
            self.indexes.append(index_path)

    def _add_vectors_in_batches(self, index, vectors):
        """
        将向量分批添加到索引中。
        
        :param index: Faiss索引对象
        :param vectors: 向量的numpy数组
        """
        n_total = vectors.shape[0]
        for start in range(0, n_total, self.batch_size):
            end = min(start + self.batch_size, n_total)
            batch_vectors = vectors[start:end]
            index.add(batch_vectors.astype('float32'))
            print(f"Added vectors {start} to {end-1}")
            
    def read_indexes(self):
        assert len(self.index_file) > 0, "No index found in the folder"
        self.indexes = []
        for index_path in self.index_file:
            index = faiss.read_index(str(index_path))
            # res = faiss.StandardGpuResources()
            # gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            # self.indexes.append(gpu_index)
            self.indexes.append(index)
        

    def search(self, query, top_k=10, docs=None):
        """
        在所有索引上执行查询并返回前top_k个结果。
        
        :param query_vector: 查询向量
        :param top_k: 返回的结果数量
        :return: 距离和索引的元组列表
        """
        all_distances = []
        all_indices = []

        query_vector = self.model.encode([query])

        assert docs is not None, "docs should not be None"

        offset = 0
        # for index_path in self.index_file:
        for index in self.indexes:
            # index = faiss.read_index(str(index_path))
            distances, indices = index.search(query_vector, top_k)
            all_distances.extend(distances[0])
            for i in indices[0]:
                all_indices.append(i + offset)
            offset += index.ntotal
        # print(len(all_distances), len(all_indices))

        sorted_results = sorted(zip(all_distances, all_indices), reverse=True, key=lambda x: x[0])
        # print(sorted_results)
        sorted_results = sorted_results[:top_k]
        res_text = []
        for i in sorted_results:
            res_text.append(docs[i[1]])
        return res_text
    
    def search_scores(self, query, top_k=10, docs=None):
        """
        在所有索引上执行查询并返回前top_k个结果文档，还有对应的分数。
        
        :param query_vector: 查询向量
        :param top_k: 返回的结果数量
        :return: 距离和索引的元组列表
        """
        all_distances = []
        all_indices = []

        query_vector = self.model.encode([query])

        assert docs is not None, "docs should not be None"

        # for index_path in self.indexes:
        for index in self.index_file:
            # index = faiss.read_index(str(index_path))
            distances, indices = index.search(query_vector, top_k)
            all_distances.extend(distances[0])
            all_indices.extend(indices[0])

        sorted_results = sorted(zip(all_distances, all_indices), key=lambda x: x[0])[:top_k]
        res_text = []
        for i in sorted_results:
            res_text.append(docs[i[1]])
        return res_text, [i[0] for i in sorted_results]


class VecRetrieval(object):
    """使用faiss进行向量检索"""

    def __init__(self, load_result, dim=1024, num=5, embedding_store='/data/chenjl/REST/embedding/embedding_index.index', model_path='BAAI/bge-m3'):
        """

        :param model_path:向量模型路径（建议使用bge模型）
        :param target_vec_file:需要写入向量库文本
        :param dim:向量维度
        :param num:返回前num个相似文本
        :param embedding_store:向量存储路径
        """
        self.model_path = model_path
        self.load_result = load_result
        self.dim = dim
        self.num = num
        self.model = SentenceModel(self.model_path)
        self.embedding_store = embedding_store
        if os.path.exists(self.embedding_store):
            self.index = faiss.read_index(self.embedding_store)
        
    def read_txt(self):
        """
        读取需要存入向量库的文件

        :return:文本列表
        """
        if hasattr(self, 'docs') is False:
            self.docs = []
            for text in self.load_result:
                if isinstance(text, str):
                    self.docs.append(text)
                elif isinstance(text, dict):
                    self.docs.append(text['text'])
        return self.docs

    def read_dict(self):
        """
        读取需要存入向量库的文件

        :return:json格式的字典
        """
        return self.load_result

    def vec_storage(self):
        """
        文本转为向量工具

        :return: 向量库对象
        """
        sentence_embedding = self.model.encode(self.read_txt())
        index = faiss.IndexFlatL2(self.dim)
        index.add(sentence_embedding)
        faiss.write_index(index, self.embedding_store)
        return index

    def vec_retrieval(self, query, rate=0.99):
        """
        向量检索；
        由于faiss向量数据库检索返回结果不为空，因此设置rate参数，计算规则比较暴力，直接取前百分之rate*100有结果，剩余的返回NULL

        :param query: 需要检索的文本，可以是字符串，也可以是列表
        :param rate:取值0-1之间，值越大，匹配到的结果越多，反之越少。匹配不到的返回NULL，自定义设置
        :return:检索到的相似文本
        """
        if isinstance(query, str):
            query_vec = self.model.encode([query], max_seq_length=64)
            if os.path.exists(self.embedding_store):
                index = faiss.read_index(self.embedding_store)
            else:
                index = self.vec_storage()
            _, I = index.search(query_vec, self.num)
            similarity_text = self.read_txt()[I[0][0]]
        elif isinstance(query, list):
            query_vec = self.model.encode(query, max_seq_length=64)
            if os.path.exists(self.embedding_store):
                start_time = time.time()
                index = faiss.read_index(self.embedding_store)
                spend_time = time.time() - start_time
                print(f'加载向量数据库耗时：{spend_time} s')
            else:
                start_time = time.time()
                index = self.vec_storage()
                spend_time = time.time() - start_time
                print(f'文本转向量耗时：{spend_time} s')
            D, I = index.search(query_vec, self.num)
            print(I)

            similarity_text = []
            dicts = self.read_dict()
            scores = [i for d in D for i in d]
            scores.sort()
            for ids, i in enumerate(I):
                # if D[ids][0] < scores[int(len(scores) * rate)]:
                #     similarity_text.append(texts[i[0]])
                # else:
                #     similarity_text.append("NULL")
                similarity_text.append(dicts[i[0]])
        else:
            similarity_text = None
        return similarity_text
    

    def vec_retrieval_topk(self, query):
        """
        向量检索；
        由于faiss向量数据库检索返回结果不为空，因此设置rate参数，计算规则比较暴力，直接取前百分之rate*100有结果，剩余的返回NULL

        :param query: 需要检索的文本，可以是字符串，也可以是列表
        :param rate:取值0-1之间，值越大，匹配到的结果越多，反之越少。匹配不到的返回NULL，自定义设置
        :return:检索到的相似文本
        """
        
        query_vec = self.model.encode([query], max_seq_length=64)
        if os.path.exists(self.embedding_store):
            start_time = time.time()
            index = self.index
            spend_time = time.time() - start_time
            # print(f'加载向量数据库耗时：{spend_time} s')
        else:
            start_time = time.time()
            index = self.vec_storage()
            spend_time = time.time() - start_time
            print(f'文本转向量耗时：{spend_time} s')
        D, I = index.search(query_vec, self.num)
        I = I[0]

        similarity_text = []
        dicts = self.read_dict()
        for i in I:
            # if D[ids][0] < scores[int(len(scores) * rate)]:
            #     similarity_text.append(texts[i[0]])
            # else:
            #     similarity_text.append("NULL")
            similarity_text.append(dicts[int(i)])
        return similarity_text
    
    
    def index_retrieval(self, queries):
        assert isinstance(queries, list), 'query needs to be list'
        query_vec = self.model.encode(queries, max_seq_length=64)
        if os.path.exists(self.embedding_store):
            start_time = time.time()
            index = self.index
            spend_time = time.time() - start_time
            print(f'加载向量数据库耗时：{spend_time} s')
        else:
            start_time = time.time()
            index = self.vec_storage()
            spend_time = time.time() - start_time
            print(f'文本转向量耗时：{spend_time} s')
        start_time = time.time()   
        D, I = index.search(query_vec, self.num)
        spend_time = time.time() - start_time
        print(f'spend_time: {spend_time}')
        return I
    
    
    def vec_retrieval_topk_scores(self, query):
        """
        向量检索；
        由于faiss向量数据库检索返回结果不为空，因此设置rate参数，计算规则比较暴力，直接取前百分之rate*100有结果，剩余的返回NULL

        :param query: 需要检索的文本，可以是字符串，也可以是列表
        :param rate:取值0-1之间，值越大，匹配到的结果越多，反之越少。匹配不到的返回NULL，自定义设置
        :return:检索到的相似文本
        """
        
        query_vec = self.model.encode([query], max_seq_length=64)
        if os.path.exists(self.embedding_store):
            start_time = time.time()
            index = self.index
            spend_time = time.time() - start_time
            # print(f'加载向量数据库耗时：{spend_time} s')
        else:
            start_time = time.time()
            index = self.vec_storage()
            spend_time = time.time() - start_time
            print(f'文本转向量耗时：{spend_time} s')
        D, I = index.search(query_vec, self.num)
        I = I[0]

        similarity_text = []
        dicts = self.read_dict()
        for i in I:
            # if D[ids][0] < scores[int(len(scores) * rate)]:
            #     similarity_text.append(texts[i[0]])
            # else:
            #     similarity_text.append("NULL")
            similarity_text.append(dicts[int(i)])
        return similarity_text, D[0]



class VecRetrieval_IVFPQ(object):
    """使用faiss进行向量检索"""

    def __init__(self, load_result, dim=1024, top_k=5, embedding_store='/data/chenjl/REST/embedding/embedding_index.index', model_path='BAAI/bge-m3'):
        """
        :param model_path: 向量模型路径（建议使用bge模型）
        :param load_result: 需要写入向量库的数据
        :param dim: 向量维度
        :param num: 返回前num个相似文本
        :param embedding_store: 向量存储路径
        """
        self.model_path = model_path
        self.load_result = load_result
        self.dim = dim
        self.top_k = top_k
        self.model = SentenceModel(self.model_path)
        self.embedding_store = embedding_store
        if os.path.exists(self.embedding_store):
            self.index = faiss.read_index(self.embedding_store)
            self.index.nprobe = 16
        else:
            quantizer = faiss.IndexFlatL2(dim)  # 使用 L2 距离度量
            self.index = faiss.IndexIVFPQ(quantizer, dim, 4096, 64, 8)
            self.index.train(self.get_train_vectors())  # 提供一些训练向量用于训练索引
            faiss.write_index(self.index, self.embedding_store)

    def read_txt(self):
        """
        读取需要存入向量库的文件

        :return: 文本列表
        """
        if not hasattr(self, 'docs'):
            self.docs = []
            for text in self.load_result:
                if isinstance(text, str):
                    self.docs.append(text)
                elif isinstance(text, dict):
                    self.docs.append(text['text'])
        return self.docs
    
    def read_dict(self):
        """
        读取需要存入向量库的文件

        :return:json格式的字典
        """
        return self.load_result

    def get_train_vectors(self):
        """
        获取一部分文本作为训练向量
        :return: 训练向量
        """
        docs = self.read_txt()
        train_size = min(len(docs), 10000)  # 取前10000条或所有文档中的最小值作为训练集
        return self.model.encode(docs[:train_size])

    def vec_storage(self, batch_size=10000):
        """
        分批次将文本转换为向量并加入索引

        :param batch_size: 每批次处理的文本数量
        :return: None
        """
        docs = self.read_txt()
        total_docs = len(docs)
        print(f"Total documents to process: {total_docs}")

        for start_idx in range(0, total_docs, batch_size):
            end_idx = min(start_idx + batch_size, total_docs)
            batch_texts = docs[start_idx:end_idx]
            batch_embeddings = self.model.encode(batch_texts)
            
            print(f"Processing batch {start_idx} to {end_idx}")
            self.index.add(batch_embeddings)

        faiss.write_index(self.index, self.embedding_store)
        print("All vectors have been added to the index.")

    def vec_retrieval(self, query, rate=0.99):
        """
        向量检索；
        由于faiss向量数据库检索返回结果不为空，因此设置rate参数，计算规则比较暴力，直接取前百分之rate*100有结果，剩余的返回NULL

        :param query: 需要检索的文本，可以是字符串，也可以是列表
        :param rate:取值0-1之间，值越大，匹配到的结果越多，反之越少。匹配不到的返回NULL，自定义设置
        :return:检索到的相似文本
        """
        if isinstance(query, str):
            query_vec = self.model.encode([query], max_seq_length=64)
            if os.path.exists(self.embedding_store):
                index = faiss.read_index(self.embedding_store)
            else:
                index = self.vec_storage()
            _, I = index.search(query_vec, self.top_k)
            similarity_text = self.read_txt()[I[0][0]]
        elif isinstance(query, list):
            query_vec = self.model.encode(query, max_seq_length=64)
            if os.path.exists(self.embedding_store):
                start_time = time.time()
                index = faiss.read_index(self.embedding_store)
                spend_time = time.time() - start_time
                print(f'加载向量数据库耗时：{spend_time} s')
            else:
                start_time = time.time()
                index = self.vec_storage()
                spend_time = time.time() - start_time
                print(f'文本转向量耗时：{spend_time} s')
            D, I = index.search(query_vec, self.top_k)
            print(I)

            similarity_text = []
            dicts = self.read_dict()
            scores = [i for d in D for i in d]
            scores.sort()
            for ids, i in enumerate(I):
                # if D[ids][0] < scores[int(len(scores) * rate)]:
                #     similarity_text.append(texts[i[0]])
                # else:
                #     similarity_text.append("NULL")
                similarity_text.append(dicts[i[0]])
        else:
            similarity_text = None
        return similarity_text
    

    def vec_retrieval_topk(self, query):
        """
        向量检索；
        由于faiss向量数据库检索返回结果不为空，因此设置rate参数，计算规则比较暴力，直接取前百分之rate*100有结果，剩余的返回NULL

        :param query: 需要检索的文本，可以是字符串，也可以是列表
        :param rate:取值0-1之间，值越大，匹配到的结果越多，反之越少。匹配不到的返回NULL，自定义设置
        :return:检索到的相似文本
        """
        
        query_vec = self.model.encode([query])
        if os.path.exists(self.embedding_store):
            start_time = time.time()
            index = self.index
            spend_time = time.time() - start_time
            # print(f'加载向量数据库耗时：{spend_time} s')
        else:
            start_time = time.time()
            index = self.vec_storage()
            spend_time = time.time() - start_time
            print(f'文本转向量耗时：{spend_time} s')
        D, I = index.search(query_vec, self.top_k)
        I = I[0]
        # print(I)

        similarity_text = []
        dicts = self.read_dict()
        for i in I:
            # if D[ids][0] < scores[int(len(scores) * rate)]:
            #     similarity_text.append(texts[i[0]])
            # else:
            #     similarity_text.append("NULL")
            similarity_text.append(dicts[int(i)])
        return similarity_text
    
    def index_retrieval(self, queries):
        assert isinstance(queries, list), 'query needs to be list'
        query_vec = self.model.encode(queries, max_seq_length=64)
        if os.path.exists(self.embedding_store):
            start_time = time.time()
            index = self.index
            spend_time = time.time() - start_time
            print(f'加载向量数据库耗时：{spend_time} s')
        else:
            start_time = time.time()
            index = self.vec_storage()
            spend_time = time.time() - start_time
            print(f'文本转向量耗时：{spend_time} s')
        start_time = time.time()   
        D, I = index.search(query_vec, self.top_k)
        spend_time = time.time() - start_time
        print(f'spend_time: {spend_time}')
        return I
