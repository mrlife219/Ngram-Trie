import os
import sys
# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

os.chdir(sys.path[0])
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# from vector_store_util import *
import torch
from contextlib import contextmanager
import numpy as np
from rest.model.rest_model import RestModel
from rest.model.kv_cache import *
from rest.model.utils import *
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
import jsonlines
torch.manual_seed(0)

from tqdm import tqdm
import time
import argparse
import json
import pickle
# from text2vec import SentenceModel, semantic_search

# 导入配置
from config import VICUNA_MODEL_PATH, LOG_FILE_DIR

torch.manual_seed(0)

def run_eval(model, tokenizer, docs, num_draft, temperature, top_p, max_new_token, top_k, n, prefix_length):
    task_results = {}
    model_name = args.model_path.split("/")[-1]
    for sample in tqdm(dataset, total=len(dataset)):
        avg_suffix_search_time = []
        avg_suffix_processing_time = []
        avg_time_per_token_list = []
        avg_search_time = []
        avg_valid_time = []
        avg_other_time = []
        avg_trie_time = []
        avg_trie_length = []
        avg_new_token = []
        avg_inference_time = 0
        avg_suffix_other_time = []

        f = 0
        accept_lengths_tree_average = []
        accept_lengths_tree_average_micro = []
        category = sample['category']
        if (sample['question_id'] >= 80 and sample['question_id'] < 160):
            category = 'mt-chat'
        # if(category not in ['summarization', 'rag']) and (sample['question_id'] <= 80 or sample['question_id'] > 160):
        if category not in ['summarization', 'rag']:
            continue
        if sample['question_id'] == 288:
            continue # too long to generate
        if "vicuna" in args.model_path:
            conv = get_conversation_template('vicuna')
        elif "llama2" in args.model_path:
            conv = get_conversation_template('llama-2')
        elif "llama3" in args.model_path:
            conv = get_conversation_template('llama-3')
        else:
            conv = get_conversation_template(args.model_path)
        for i in range(len(sample['turns'])):
            qs = sample['turns'][i]
            end_flag = 0

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
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                if 'vicuna' or 'llama2' in args.model_path:
                    conv.stop_str = "</s>"
                else:
                    conv.stop_str = "<|eot_id|>"
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids
                # print(qs_tokens)
                input_len = len(input_ids[0])
                input_ids = torch.as_tensor(input_ids).cuda()
                model.base_model.model.draft_mask = None
                # outputs = model.base_model(input_ids, past_key_values = past_key_values, use_cache=True)
                new_token = 0
                logits = initialize_logits(
                        input_ids, model, past_key_values
                )
                cur_length = input_len + 1
                accept_lengths_tree.append(1)
                start_time = time.time()
                
                torch.cuda.synchronize()
                if category in ['qa']:
                    pass
                else:
                    start_search_time = time.time()
                    match_trie = trie_generate([qs], tokenizer=tokenizer, freq_threshold=1, n=n, prefix_length=prefix_length)
                    prepare_time = time.time() - start_search_time
                    avg_trie_time.append(prepare_time)
                    avg_trie_length.append(len(match_trie))

                
                if end_flag == 0:
                    # logits = outputs.logits
                    cur_length = input_len + 1

                    
                    search_time = 0
                    valid_time = 0
                    other_time = 0
                    for j in range(2000):
                        start_search_time = time.time()
                        debug = True
                        candidates, tree_candidates, draft_buffers, suffix_search_time, suffix_process_time, suffix_other_time = generate_candidates_and_draft_buffer_rag(
                                logits,
                                input_ids,
                                match_trie,
                                top_p,
                                temperature,
                                num_draft=num_draft,
                                device=model.base_model.device,
                                debug=debug
                            )
                        # print(f'search time: {time.time() - start_search_time}')
                        # print(candidates)
                        search_time += time.time() - start_search_time
                        
                        model.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]
                        
                        start_valid_time = time.time()

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
                        valid_time += time.time() - start_valid_time
                        start_other_time = time.time()
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
                        # if j == 0:
                        #     print(tokenizer.decode(input_ids[0, cur_length-1]), end=' ')
                        other_time += time.time() - start_other_time
                        accept_length_tree = input_ids.shape[1] - cur_length
                        # if accept_length_tree == 1:
                        #     print(tokenizer.decode(input_ids[0, cur_length:]), end=' ')
                        # else:
                        #     print(f"\033[1;33m{tokenizer.decode(input_ids[0, cur_length:])}\033[0m", end=' ')
                        cur_length = accept_length_tree + cur_length
                        accept_lengths_tree.append(accept_length_tree)
                        
                        avg_suffix_search_time.append(suffix_search_time)
                        if suffix_process_time > 0:
                            avg_suffix_processing_time.append(suffix_process_time)
                        avg_suffix_other_time.append(suffix_other_time)
                        if model.tokenizer.eos_token_id in input_ids[0, input_len:] or new_token > max_new_token:
                            new_token = new_token.cpu()
                            break
                

                # if f == 0:  
                #     print(tokenizer.batch_decode(input_ids))
                #     f += 1
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                avg_new_token.append(new_token)
                avg_inference_time += total_time
                avg_time_per_token = total_time / new_token
                avg_time_per_token_list.append(avg_time_per_token)
                f += 1
                match_trie.clear()
                if len(accept_lengths_tree) > 0:
                    accept_lengths_tree_average.append(np.mean(accept_lengths_tree))
                    accept_lengths_tree_average_micro.extend(accept_lengths_tree)
                    avg_search_time.append(search_time)
                    avg_valid_time.append(valid_time)
                    avg_other_time.append(other_time)

                # 使用配置文件中的路径构建日志文件路径
                log_file = os.path.join(LOG_FILE_DIR, f"{model_name}_sdbench.jsonl")
                with jsonlines.open(log_file, mode='a') as writer:
                    writer.write({
                        "question_id": sample['question_id'],
                        "category": category,
                        "output": tokenizer.decode(input_ids[0, input_len:]),
                        "accept_length_tree": accept_lengths_tree,
                    })
            
            task_results[category] = {
                "avg_time_per_token": np.mean(avg_time_per_token_list),
                "avg_token_per_second": 1 / np.mean(avg_time_per_token_list),
                # "avg_Rouge_L": np.mean(avg_Rouge_L),
                "avg_inference_time": avg_inference_time / len(sample['turns']),
                "avg_trie_time": np.mean(avg_trie_time),
                "avg_trie_length": np.mean(avg_trie_length),
                "avg_new_token": np.mean(avg_new_token),
                "accept_lengths_tree_average": np.mean(accept_lengths_tree_average),
                "accept_lengths_tree_average_micro": np.mean(accept_lengths_tree_average_micro),
            }
    for key, value in task_results.items():
        print(f"{key}:")
        print(f"avg_time_per_token: {value['avg_time_per_token']}")
        print(f"avg_token_per_second: {value['avg_token_per_second']}")
        # print(f"avg_Rouge_L: {value['avg_Rouge_L']}")
        print(f"avg_inference_time: {value['avg_inference_time']}")
        print(f"avg_trie_time: {value['avg_trie_time']}")
        print(f"avg_trie_length: {value['avg_trie_length']}")
        print(f"avg_new_token: {value['avg_new_token']}")
        print(f"accept_lengths_tree_average: {value['accept_lengths_tree_average']}")
        print(f"accept_lengths_tree_average_micro: {value['accept_lengths_tree_average_micro']}")
        print(f"avg_search_time: {np.mean(avg_search_time)}")
        print(f"avg_valid_time: {np.mean(avg_valid_time)}")
        print(f"avg_other_time: {np.mean(avg_other_time)}")
        print(f"avg_suffix_search_time: {np.mean(avg_suffix_search_time)}")
        print(f"avg_suffix_processing_time: {np.mean(avg_suffix_processing_time)}")
        print(f"avg_suffix_other_time: {np.mean(avg_suffix_other_time)}")
        print("*"*30)
        print()
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=VICUNA_MODEL_PATH,  # 使用配置文件中的路径
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

    dataset = load_questions(args.dataset_path, begin=None, end=None)
    # print(dataset[0])
    # datastore = pickle.load(open(SHAREGPT_NGRAM_PKL_PATH, 'rb'))  # 使用配置文件中的路径
    # datastore = pickle.load(open('/data/chenjl/REST/ultrachat_splitted.pkl', 'rb'))
    print(len(dataset))
    # docs = datastore
    docs = []
    # trie = pickle.load(open(SHAREGPT_NGRAM_PKL_PATH, 'rb'))  # 使用配置文件中的路径

    # for n in range(8, 17):
    #     for prefix_length in range(2, 6):
    #         print(f"n: {n}, prefix_length: {prefix_length}")
    #         run_eval(
    #             model, 
    #             tokenizer, 
    #             docs, 
    #             args.num_draft,
    #             args.temperature, 
    #             args.top_p,
    #             args.max_new_token,
    #             args.top_k,
    #             n,
    #             prefix_length,
    #         )
    n = 13
    prefix_length = 3
    run_eval(
        model, 
        tokenizer, 
        docs, 
        args.num_draft,
        args.temperature, 
        args.top_p,
        args.max_new_token,
        args.top_k,
        n,
        prefix_length,
    )