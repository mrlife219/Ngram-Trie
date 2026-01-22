import os
import sys
# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

os.chdir(sys.path[0])
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from vector_store_util import *
import torch
from contextlib import contextmanager
import numpy as np
from rest.model.rest_model import RestModel
from rest.model.kv_cache import *
from rest.model.utils import *
from langchain.text_splitter import TokenTextSplitter
import random
import pickle
random.seed(42)

from tqdm import tqdm
import time
import argparse
import jsonlines
from fastchat.model import get_conversation_template

# 导入配置
from config import VICUNA_MODEL_PATH, LLAMA2_MODEL_PATH, LLAMA3_MODEL_PATH, LOG_FILE_DIR, WIKIPEDIA_DPR_IVFPQ_INDEX_PATH, WIKIPEDIA_DPR_SPLITTED_PKL_PATH

def generate_opendomainqa_prompt(question, context):
    """
    Generates an Open-Domain QA prompt based on provided question and context.
    
    :param question: The specific question to be answered.
    :param context: A list of strings where each string represents a document's content.
    :return: A formatted prompt for open-domain question answering.
    """
    prompt = ""
    
    for doc in context:
        prompt += f"{doc}\n\n"
        
    prompt += f"Based on the documents above, please answer the question '{question}'. Ensure your answer is accurate and comprehensive by carefully considering all provided documents. If discrepancies or updates exist among documents, prioritize more recent information and explain your reasoning.\n\n"
    
    return prompt

def run_eval(model, tokenizer, docs, num_draft, temperature, top_p, max_new_token, top_k):
    model_name = args.model_path.split('/')[-1]
    avg_time_per_token_list = []

    avg_trie_time = []
    avg_trie_length = []
    avg_new_token = []
    avg_Rouge_L = []
    avg_inference_time = 0

    f = 0
    accept_lengths_tree_average = []
    accept_lengths_tree_average_micro = []

    dataset = jsonlines.open(args.dataset_path)
    for sample in tqdm(dataset, total=100):
        prompt = sample['turns'][0]
        # print(prompt)

        accept_lengths_tree = []
        with torch.inference_mode():
            # past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
            # model.past_key_values = past_key_values
            # model.past_key_values_data = past_key_values_data
            # model.current_length_data = current_length_data

            # model.current_length_data.zero_() # this is for rerun

            # Initialize the past key and value states
            if hasattr(model, "past_key_values"):
                past_key_values = model.past_key_values
                past_key_values_data = model.past_key_values_data
                current_length_data = model.current_length_data
                # Reset the past key and value states
                current_length_data.zero_()
            else:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(model.base_model)
                model.past_key_values = past_key_values
                model.past_key_values_data = past_key_values_data
                model.current_length_data = current_length_data

            
            match_docs = vec_test.vec_retrieval_topk(prompt)
            match_texts = [match_doc['text'] for match_doc in match_docs]
            prompt = generate_opendomainqa_prompt(question=prompt, context=match_texts)
            
            if args.model_path == VICUNA_MODEL_PATH:  # 使用配置文件中的路径
                conv = get_conversation_template('vicuna')
            elif args.model_path == LLAMA2_MODEL_PATH:  # 使用配置文件中的路径
                conv = get_conversation_template('llama-2')
            elif args.model_path == LLAMA3_MODEL_PATH:  # 使用配置文件中的路径
                conv = get_conversation_template('llama-3')
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            if "vicuna" in args.model_path or "llama2" in args.model_path:
                conv.stop_str = "</s>"
            elif "llama3" in args.model_path:
                conv.stop_str = "<|eot_id|>"
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids
            input_len = len(input_ids[0])
            # print(input_len)
            input_ids = torch.as_tensor(input_ids).cuda()
            model.base_model.model.draft_mask = None
            # outputs = model.base_model(input_ids, use_cache=True)
            new_token = 0
            logits = initialize_logits(
                    input_ids, model, past_key_values
            )
            cur_length = input_len + 1
            accept_lengths_tree.append(1)
            
            torch.cuda.synchronize()
            start_time = time.time()
            match_trie = trie_generate(docs=match_texts, tokenizer=tokenizer, n=13, prefix_length=3)
            # match_trie = trie
            prepare_time = time.time() - start_time
            avg_trie_time.append(prepare_time)
            avg_trie_length.append(len(match_trie))


            try:
                for i in range(2000):
                    # start_search_time = time.time()
                    candidates, tree_candidates, draft_buffers = generate_candidates_and_draft_buffer_rag(
                            logits,
                            input_ids,
                            match_trie,
                            top_p,
                            temperature,
                            num_draft=num_draft,
                            device=model.base_model.device
                        )
                    # print(f'search time: {time.time() - start_search_time}')
                    # print(candidates)
                    
                    model.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]

                    logits, outputs = tree_decoding(
                            model,
                            tree_candidates,
                            past_key_values,
                            draft_buffers["draft_position_ids"],
                            input_ids,
                            draft_buffers["retrieve_indices"],
                        )

                    best_candidate, accept_length = evaluate_posterior(
                            logits, candidates, temperature = temperature, top_p=top_p
                        )
                    input_ids, logits, new_token = update_inference_inputs(
                            input_ids,
                            candidates,
                            best_candidate,
                            accept_length,
                            draft_buffers["retrieve_indices"],
                            outputs,
                            logits,
                            new_token,
                            past_key_values_data,
                            current_length_data,
                        )
                    accept_length_tree = input_ids.shape[1] - cur_length
                    cur_length = accept_length_tree + cur_length
                    accept_lengths_tree.append(accept_length_tree)
                    if model.tokenizer.eos_token_id in input_ids[0, input_len:] or new_token > max_new_token:
                        new_token = new_token.cpu()
                        break
            except Exception as e:
                print(e, sample['question_id'])
                continue
            

            # if f == 0:  
            #     print(tokenizer.batch_decode(input_ids))
            #     f += 1
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            avg_inference_time += total_time
            avg_new_token.append(new_token)
            avg_time_per_token = total_time / new_token
            avg_time_per_token_list.append(avg_time_per_token)
            
            f += 1
            if len(accept_lengths_tree) > 0:
                accept_lengths_tree_average.append(np.mean(accept_lengths_tree))
                accept_lengths_tree_average_micro.extend(accept_lengths_tree)
            # 使用配置文件中的路径构建日志文件路径
            log_file = os.path.join(LOG_FILE_DIR, f"task_trivia_ablation_log_model_{model_name}.jsonl")
            with jsonlines.open(log_file, mode='a') as writer:
                writer.write({
                    "question_id": sample['question_id'],
                    "category": "hagrid",
                    "output": tokenizer.decode(input_ids[0, input_len:]),
                    "accept_lengths_tree": accept_lengths_tree,
                    'trie_size': len(match_trie),
                    'new_token': new_token.tolist(),
                })
                match_trie.clear()

    print("accept_lengths_tree_average: ", np.mean(accept_lengths_tree_average))
    print("accept_lengths_tree_average_micro: ", np.mean(accept_lengths_tree_average_micro))
    # print("avg_inference_time: ", avg_inference_time / f)
    print("avg_token_per_second: ", 1 / np.mean(avg_time_per_token_list))
    # print("avg Rouge L: ", np.mean(avg_Rouge_L))
    # print("avg_time_per_token_micro: ", np.sum([item[0] for item in avg_time_per_token_list_micro]) / np.sum([item[1] for item in avg_time_per_token_list_micro]))
    print("avg time in trie preparing: ", np.mean(avg_trie_time))
    print("avg trie length: ", np.mean(avg_trie_length))
    print('avg_new_token: ', np.mean(avg_new_token))
    print("*"*30)
    print()

    log_dict = {
        'accept_lengths_tree_average': np.mean(accept_lengths_tree_average),
        'accept_lengths_tree_average_micro': np.mean(accept_lengths_tree_average_micro),
        'avg_inference_time': avg_inference_time / f,
        'avg_time_per_token': 1 / np.mean(avg_time_per_token_list),
        'avg_Rouge_L': np.mean(avg_Rouge_L),
        'avg_trie_time': np.mean(avg_trie_time),
        'avg_trie_length': np.mean(avg_trie_length),
        'avg_new_token': np.mean(avg_new_token),
    }
    with open(f"log_vicuna", 'a') as f:
        f.write(f"top_k: {top_k}\n")
        f.write(str(log_dict))
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=LLAMA2_MODEL_PATH,  # 使用配置文件中的路径
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        # 默认路径需要根据实际需求设置
        help="The path to the HumanEval dataset",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="The threshold for nucleus sampling.",
    )

    parser.add_argument(
        "--num-draft",
        type=int,
        default=64,
        help="The number of draft tokens.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="The most relevant documents chunk numbers to search",
    )

    args = parser.parse_args()

    if args.temperature == 0:
        args.top_p = 0
        
    print(args)

    model = RestModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
   
    tokenizer = model.get_tokenizer()
    
    datastore = []
    
    docs = pickle.load(open(WIKIPEDIA_DPR_SPLITTED_PKL_PATH, 'rb'))  # 使用配置文件中的路径
    
    vec_test = VecRetrieval_IVFPQ(load_result=docs, top_k=args.top_k, embedding_store=WIKIPEDIA_DPR_IVFPQ_INDEX_PATH)  # 使用配置文件中的路径

    run_eval(
        model, 
        tokenizer, 
        datastore, 
        args.num_draft,
        args.temperature, 
        args.top_p,
        args.max_new_token,
        args.top_k
    )
    
    # run_eval(
    #         model, 
    #         tokenizer, 
    #         datastore, 
    #         args.num_draft,
    #         args.temperature, 
    #         args.top_p,
    #         args.max_new_token,
    #         args.top_k
    #     )