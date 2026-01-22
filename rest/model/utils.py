# adapted from https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/utils.py

from __future__ import print_function
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import numpy as np
import torch
import torch.nn.functional as F

import time
from collections import defaultdict
import heapq
from collections import Counter
import bm25s
import string
from collections import defaultdict

class ArrayTrie:
    def __init__(self):
        self.prefix_map = defaultdict(list)  # 存储所有前缀路径的值
        self.value_store = {}                # ID到值的映射
        self._counter = 0

    def __len__(self) -> int:
        return self._counter

    def insert(self, key: list, value: object) -> None:
        """修改后的插入方法：允许重复键，每次插入都记录到所有路径节点"""
        self._counter += 1
        uid = self._counter
        
        # 为所有前缀路径添加当前值的引用
        for i in range(1, len(key)+1):
            self.prefix_map[tuple(key[:i])].append(uid)
        
        self.value_store[uid] = value

    def search_longest_prefix(self, key: list) -> list:
        """精确匹配原始Trie行为的查询方法"""
        # 反向查找最长前缀
        for i in range(len(key), 0, -1):
            prefix = tuple(key[:i])
            if prefix in self.prefix_map:
                # 返回该前缀节点存储的所有有效值
                return [self.value_store[uid] 
                        for uid in self.prefix_map[prefix] 
                        if uid in self.value_store]
        return None

    def clear(self) -> None:
        """清空方法保持内存高效"""
        self.prefix_map.clear()
        self.value_store.clear()
        self._counter = 0


noisy_string = """
Theres no doubt that my mother gives all her love to me. I do believe she is a great person who makes my life beautiful and meaningful.
She is an easygoing and kind woman with bright eyes and a lovely smile. Although she is often busy, I still feel that I am taken good care of by her. 
Its a great pleasure to chat with her when I get into troubles. She always encourages me not to give up and tries to cheer me up by coming up with good solutions. In addition, I am fascinated by her cooking and writing.
With her love, I feel like a fish swimming happily in a beautiful sea. Ill cherish her love forever.
My teacher, Miss Wang, helped me a lot in my middle school life. She is a kind easygoing woman. I must thank her for making a confident girl.
I used to be a shy and unconfident girl. Mrs. Wang noticed me. She took good care of me and encouraged me to join the school speech contest. Of course, I failed. But Mrs. Wang cheered me up and said every man is the architect of his own future.
From then on, I practiced every day. It goes without saying “No pain, no gain.” I won the contest in the second term. In my opinion, teachers are the same as gardeners and they volunteer today and gain tomorrow. 
Not only can they teach knowledge but also they can teach students how to be a successful man.
Thank you, Mrs. Wang. You make a duck become a beautiful swan. I want to be a teacher that as same as you in the future.
"""
has_run = False


def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))



def initialize_logits(input_ids, model, past_key_values):
    """
    Forward pass through the model to obtain the model outputs, and logits.


    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model: The LLM for generation.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - logits (torch.Tensor): logits from the LLM.
    """
    outputs, logits = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )
    return logits


def reset_past_key_values(passed_key_values):
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates_and_draft_buffer(logits, input_ids, datastore, token_spans, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - logits (torch.Tensor): Original logits.
    - tree_indices (list or torch.Tensor): Indices associated with a tree structure.
    - retrieve_indices (list or torch.Tensor): Indices for retrieving candidates.
    
    Returns:
    - tuple: Returns cartesian candidates and tree candidates.
    """

    # Greedy decoding: Select the most probable candidate from the original logits.
    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)
        
    retrieved_token_list = []
    _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = [], [], [], []
    for span_id, token_span in enumerate(token_spans):
        this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()
        # Retrieve draft tokens from the datastore, and get draft buffer
        retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = datastore.search(this_token, choices=max_num_draft)
    
        # No retrieved sequences
        if len(retrieved_token_list) == 0:
            continue
        # Break because this span has hitted
        else:
            break
    # TODO: just continue to the next retrieval process
    if len(retrieved_token_list) == 0:
        # Just randomlt guess one token
        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]
    else:
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates, draft_buffers


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    draft_position_ids,
    input_ids,
    retrieve_indices,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    
    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - draft_position_ids (torch.Tensor): Positional IDs (Layer IDs in the Trie) of each draft token.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.
    
    Returns:
    - tuple: Returns logits, and other outputs from the model.
    """

    # Compute new position IDs by adding the draft position IDs to the length of the input sequence.
    position_ids = draft_position_ids + input_ids.shape[1]

    # Use the model to decode the tree candidates. 
    # The model is expected to return each draft token's logits, and possibly other outputs.
    outputs, tree_logits = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    
    # Reorder the obtained logits based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = tree_logits[0, retrieve_indices]

    return logits, outputs

def get_nucleus_posterior_mask(logits, candidates, temperature, top_p):

    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :-1] / temperature

    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)

    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)


    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')

    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask


def evaluate_posterior(
    logits, candidates, temperature, top_p=0.8
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    elif top_p > 0:
        assert top_p < 1.0, "top_p should between 0 and 1"
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    else:
        raise NotImplementedError


def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    outputs,
    logits,
    new_token,
    past_key_values_data,
    current_length_data,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - outputs, logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - new_token (int): Updated counter for the new tokens added.
    """
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1]], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    tgt = past_key_values_data[..., select_indices, :]
    # Destination tensor where the relevant past information will be stored
    dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
    # Copy relevant past information from the source to the destination
    dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # Extract logits for the accepted tokens
    logits = logits[None, best_candidate, accept_length : accept_length + 1]

    # Update the new token counter
    new_token += accept_length + 1

    return input_ids, logits, new_token


def top_p_filtering(logits, top_p=0.0, filter_value=float('-inf')):
    # from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79


    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits


def trie_generate(docs=None, tokenizer=None, max_results=0, freq_threshold=0, input_ids=None, mode="doc", n=10, prefix_length=3):
    """
    generate a trie for retrieval (10-gram token sequences)
    params:
    - docs: the external document. In the demo, it means a code dataset.
    - tokenizer: the LLM for tokenize.
    - max_results: the max number of results to return.
    - freq_threshold: the frequency threshold for the n-gram.
    - input_ids: the input_ids for the current step.
    - mode: the mode for the current step. Use 'doc' for inputting documents and 'token' for inputting token sequences.

    Returns:
    - results: A 2-dimensional 10-gram
    """
    # if freq_threshold == 0:
    #     assert max_results != 0, "Max_results and freq_threshold cannot all be zero."
    #     f = 0
    # else:
    #     f = 1
    eos_token_id = tokenizer.eos_token_id
    # n = 13
    ngram = defaultdict(int)
    # prefix_length = 3  # 修改为前3
    suffix_length = n - prefix_length  
    
    if mode == "doc":
        input_ids = tokenizer(docs, add_special_tokens=False)["input_ids"]

    # noisy_ids = tokenizer(noisy_string, add_special_tokens=False)["input_ids"]
    # input_ids[0].extend(noisy_ids)
    # global has_run
    # if has_run == False:
    #     print(len(noisy_ids))
    #     has_run = True
    
    for input_id in input_ids:
        input_id.append(eos_token_id)
        for k in range(suffix_length+1, n):  # 调整循环范围以适应新的前缀长度
            if len(input_id[:k]) == k:
                key = tuple(input_id[:k])
                ngram[key] += 1
        
        for k in range(len(input_id) - n + 1):
            key = tuple(input_id[k: k + n])
            ngram[key] += 1
    
    results = []
    for key, value in ngram.items():
        num_list = list(key)
        if len(num_list) < n:
            for k in range(1, prefix_length+1):
                if len(num_list) - k > 1 and ((suffix_length + k) - len(num_list)) >= 0:
                    padded_num_list = (prefix_length - k) * [eos_token_id] + num_list + ((suffix_length + k) - len(num_list)) * [eos_token_id]
                    assert len(padded_num_list) == n, f"padded_num_list length error: {len(padded_num_list)}, {padded_num_list}"
                    results.append((padded_num_list, value))
        else:
            results.append((num_list, value))
    
    # import cProfile
    # profiler = cProfile.Profile()
    # profiler.enable()
    trie = ArrayTrie()
    for dict_item in results:
        kv = dict_item[0]
        freq = dict_item[1]
        key = list(reversed(kv[:prefix_length]))
        value = kv[prefix_length:]
        trie.insert(key, (value, freq))
        # print(f"trie insert time: {time.time() - start_trietime:.2f}s")
    # profiler.disable()
    
    # trie_time = time.time() - start_time
    # if trie_time > 0.1:
    #     profiler.print_stats(sort="tottime")
    
    return trie



# def trie_generate(docs=None, tokenizer=None, max_results=0, freq_threshold=0, input_ids=None, mode="doc"):
#     eos_token_id = tokenizer.eos_token_id
#     ngram = defaultdict(int)
#     n = 10

#     if mode == "doc":
#         input_ids = [tokenizer(doc, add_special_tokens=False)["input_ids"] for doc in docs]
#     else:
#         input_ids = input_ids if input_ids else []

#     # 生成 n-gram（元组作为键）
#     for seq in input_ids:
#         seq = seq + [eos_token_id]
#         seq_len = len(seq)
        
#         # 生成前缀片段（n-3 到 n-gram）
#         for k in range(max(n-3, 1), min(n+1, seq_len+1)):
#             ngram[tuple(seq[:k])] += 1
        
#         # 滑动窗口生成 10-gram
#         for i in range(seq_len - n + 1):
#             ngram[tuple(seq[i:i+n])] += 1

#     # 填充并生成结果
#     results = []
#     for gram, freq in ngram.items():
#         gram_len = len(gram)
#         if gram_len < 10:
#             pad_front = max(4 - gram_len, 0)
#             pad_back = 10 - gram_len - pad_front
#             padded = [eos_token_id] * pad_front + list(gram) + [eos_token_id] * pad_back
#         else:
#             padded = list(gram)
#         results.append((padded, freq))

#     # 构建 Trie
#     trie = pygtrie.StringTrie(separator=",")
#     for tokens, freq in results:
#         prefix = list(reversed(tokens[:4]))
#         suffix = tokens[4:]
#         for i in range(4):
#             current_prefix = prefix[:i+1]
#             key = ",".join(map(str, current_prefix))
#             trie.setdefault(key, []).append((suffix, freq))
    
#     return trie



def get_draft_choices(paths):
    path_dict = defaultdict(dict)
    cnt_dict = defaultdict(int)
    
    max_depth = max(len(path) for path in paths)

    for depth in range(max_depth):
        cnt_dict[depth] = 0

    for path in paths:
        for depth, item in enumerate(path):
            if item not in path_dict[depth]:
                current_cnt = cnt_dict[depth]
                path_dict[depth][item] = current_cnt
                cnt_dict[depth] += 1

    max_branch = max(len(v) for v in path_dict.values())

    draft_choices = set()
    for path in paths:
        for depth in range(len(path)):
            draft_choice = [path_dict[prev_depth][path[prev_depth]] for prev_depth in range(depth + 1)]
            draft_choices.add(tuple(draft_choice))

    draft_choices = [list(choice) for choice in draft_choices]
    
    return draft_choices, max_branch


def cut_to_choices(paths, choices):
    count = [(len(set(p)), i) for i, p in enumerate(paths)]
    count.sort(key=lambda x: x[0], reverse=True)

    total_unique = sum(x for x, _ in count)
    to_remove = []

    for c, i in count:
        if total_unique > choices:
            total_unique -= c
            to_remove.append(i)
        else:
            break

    return [p for i, p in enumerate(paths) if i not in to_remove]


def generate_draft_buffers(draft_choices, topk):
    # 对草稿选择按长度排序
    sorted_draft_choices = sorted(draft_choices, key=lambda x: (len(x), x))
    
    draft_len = len(sorted_draft_choices) + 1
    # assert draft_len <= 65, "draft_len should not exceed 65"

    # 初始化深度计数
    depth_counts = [0] * draft_len
    prev_depth = 0
    for path in sorted_draft_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts[depth - 1] = 0
        depth_counts[depth - 1] += 1
        prev_depth = depth

    # 创建草稿的注意力掩码
    draft_attn_mask = [[0] * draft_len for _ in range(draft_len)]
    for i in range(draft_len):
        draft_attn_mask[i][0] = 1
        draft_attn_mask[i][i] = 1

    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_draft_choice = sorted_draft_choices[start + j]
            if len(cur_draft_choice) == 1:
                continue
            
            ancestor_idx = []
            for c in range(len(cur_draft_choice) - 1):
                index = next((idx for idx, x in enumerate(sorted_draft_choices) if x[:c+1] == cur_draft_choice[:c+1]), None)
                ancestor_idx.append(index + 1)

            for idx in ancestor_idx:
                draft_attn_mask[j + start + 1][idx] = 1
        start += depth_counts[i]

    # 生成草稿树索引
    draft_tree_indices = [0] * draft_len
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_draft_choice = sorted_draft_choices[start + j]
            draft_tree_indices[start + j + 1] = cur_draft_choice[-1] + topk * i + 1
        start += depth_counts[i]

    # 生成位置ID
    draft_position_ids = [0] * draft_len
    start = 0
    for i in range(len(depth_counts)):
        for j in range(start + 1, start + depth_counts[i] + 1):
            draft_position_ids[j] = i + 1
        start += depth_counts[i]

    # 生成检索索引
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_draft_choices)):
        cur_draft_choice = sorted_draft_choices[len(sorted_draft_choices) - 1 - i]
        retrieve_indice = []
        if cur_draft_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_draft_choice)):
                index = sorted_draft_choices.index(cur_draft_choice[:c + 1])
                retrieve_indice.append(index)
                retrieve_paths.append(cur_draft_choice[:c + 1])
        retrieve_indices_nest.append(retrieve_indice)

    max_length = max(len(x) for x in retrieve_indices_nest)
    retrieve_indices = [pad_path(x, max_length, -2) for x in retrieve_indices_nest]

    for i in range(len(retrieve_indices)):
        for j in range(len(retrieve_indices[i])):
            retrieve_indices[i][j] += 1

    for i in range(len(retrieve_indices)):
        retrieve_indices[i].insert(0, 0)

    return draft_attn_mask, draft_tree_indices, draft_position_ids, retrieve_indices


def to_torchtensor(list):
    return torch.tensor(list, dtype=torch.int32)


def generate_candidates_and_draft_buffer_rag(
                        logits,
                        input_ids,
                        trie,
                        top_p,
                        temperature,
                        num_draft=64,
                        device="cuda",
                        debug=False):
    if debug:
        start_time = time.time()

    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)

    suffix = retrieve_by_ngram(trie, input_ids_extend.squeeze(0))
    if debug:
        suffix_search_time = time.time() - start_time

    # print(f"suffix search time: {suffix_search_time:.2f}s")

    if suffix is None:
        suffix_process_time = 0
        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]

    else:
        # cnt_map = {}
        # for token_tuple in suffix:
        #     retrieved_token = token_tuple[0]
        #     for j in range(len(retrieved_token)):
        #         tmp_token = retrieved_token[:j + 1]
        #         cnt_map[tuple(tmp_token)] = cnt_map.get(tuple(tmp_token), 0) + token_tuple[1]

        # heap = []
        # for k, v in cnt_map.items():
        #     if len(heap) < num_draft:
        #         heapq.heappush(heap, (v, k))
        #     elif v > heap[0][0]:
        #         heapq.heappop(heap)  # 将最小的弹出
        #         heapq.heappush(heap, (v, k))

        # verified = [k for _, k in heap]
        # verified = list(set(tuple(v) for v in verified))
        # verified = [list(v) for v in verified]
        # paths = cut_to_choices(verified, num_draft)
        if debug:
            start_process_time = time.time()
        if len(suffix)*len(suffix[0][0]) > num_draft:
            cnt_map = {}
            for token_tuple in suffix:
                retrieved_token, weight = token_tuple[0], token_tuple[1]  # 假设结构为 (token_list, weight)
                for j in range(len(retrieved_token)):
                    tmp_token = retrieved_token[:j+1]
                    key = tuple(tmp_token)
                    cnt_map[key] = cnt_map.get(key, 0) + weight  # 累加权值

            # 直接获取前num_draft个高频元素，省略去重步骤
            verified = heapq.nlargest(num_draft, cnt_map.items(), key=lambda item: item[1])
            verified = [list(k) for k, _ in verified]  # 直接提取键，无需去重

            paths = cut_to_choices(verified, num_draft)
        else:
            paths = [suf[0] for suf in suffix]
        

        draft_choices, max_branch = get_draft_choices(paths)
        
        max_length = max((len(path) for path in paths), default=0)
        padded_paths = [pad_path(path, max_length, -2) for path in paths]

        retrieved_token_list = padded_paths
        # print(draft_choices)
        # print(max_branch)
        _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = generate_draft_buffers(draft_choices, max_branch)

    
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        if debug:
            suffix_process_time = time.time() - start_process_time
        
    if debug:
        start_other_time = time.time()
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()
    # print(candidates_draft_logits)

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)
    # print(candidates.shape)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)

    if debug:
        suffix_other_time = time.time() - start_other_time
        return cart_candidates, tree_candidates, draft_buffers, suffix_search_time, suffix_process_time, suffix_other_time
    
    return cart_candidates, tree_candidates, draft_buffers



# def rouge_L_score(text1, text2):
#     from rouge import Rouge
#     rouge = Rouge()
#     rouge_score = rouge.get_scores(text1, text2)
#     return rouge_score[0]["rouge-l"]['f']


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))



def f1_score(prediction, ground_truth):
    # 首先把prediction和ground_truth标准化（即用上面的函数进行处理）
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    # 统计他们共有的字符
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    # 计算共有的字符的总量
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    # 计算precision
    precision = 1.0 * num_same / len(prediction_tokens)
    # 计算recall
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def bm25_search(docs, query, top_k=10):
    if len(docs) < top_k:
        top_k = len(docs)
    corpus_tokens = bm25s.tokenize(docs)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    query_tokens = bm25s.tokenize(query)
    results, scores = retriever.retrieve(query_tokens, corpus=docs, k=top_k)
    return results[0].tolist()



def retrieve_by_ngram(trie, input_ids: torch.Tensor, choices=32):
    """
    retrieve the trie to find suffix in prefix of input_id[-4:]
    """
    temp = input_ids.tolist()[-4:]
    temp = list(reversed(temp))
    # prefix = ','.join(str(i) for i in temp)
    # suffix = trie.longest_prefix(prefix).value
    suffix = trie.search_longest_prefix(temp)
    return suffix


def bm25_search_by_local_index(retriever, docs, query, top_k=10):
    if len(docs) < top_k:
        top_k = len(docs)
    query_tokens = bm25s.tokenize(query)
    results, scores = retriever.retrieve(query_tokens, corpus=docs, k=top_k)
    # print('scores:', scores)
    return results[0].tolist()


def bm25_search_score(retriever, query, top_k=10):
    # if len(docs) < top_k:
    #     top_k = len(docs)
    query_tokens = bm25s.tokenize(query)
    results, scores = retriever.retrieve(query_tokens, k=top_k)
    return results[0].tolist(), scores[0].tolist()