# Faster In-Context Learning for LLMs via N-Gram Trie Speculative Decoding

This repository contains the code and evaluation scripts for the paper "Faster In-Context Learning for LLMs via N-Gram Trie Speculative Decoding". The paper presents a novel approach to improve the efficiency of in-context learning for large language models (LLMs) by leveraging n-gram trie speculative decoding.

## Supported Models
- Llama series (Llama2, Llama3)
- Qwen series (Alibaba's Qwen models)

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Note: This project uses specific versions of libraries for compatibility. Ensure you have Python >= 3.9 and compatible CUDA drivers if using GPU acceleration.

2. For Trivial Benchmark, we use `wikipedia_dpr_preprocess.py` for preprocessing. The initial preprocessing step is included in the `wikipedia_dpr_preprocess.py` script.
  You may need to modify the script according to your specific needs. For example: 
  ```python
  with open('your_wikipedia_file.csv', 'r') as f:
  ```
3. Other datasets doesn't need preprocessing.

## Usage
Use the provided shell scripts:
- `eval_llama2.sh`: Evaluate Llama2 models
- `eval_llama3.sh`: Evaluate Llama3 models  
- `eval_qwen.sh`: Evaluate Qwen models

## Evaluation Scripts
These shell scripts automate the evaluation process for different model families:
- Configure model paths and parameters
- Run inference on benchmark datasets
- Generate performance metrics and logs

Example usage:
```bash
bash eval_llama2.sh
```

## Baselines
Baselines may be found in [Spec-Bench](https://github.com/hemingkx/Spec-Bench).

## Acknowledgments
This codebase is built from [REST](https://github.com/FasterDecoding/REST). We sincerely thank the authors for their work. And this code is influenced by remarkable projects from the LLM community.

## Citation
```
@inproceedings{chen-etal-2025-faster,
    title = "Faster In-Context Learning for {LLM}s via N-Gram Trie Speculative Decoding",
    author = "Chen, Jinglin  and
      Li, Qiwei  and
      Li, Zuchao  and
      Qi, Baoyuan  and
      Guoming, Liu  and
      Ai, Haojun  and
      Zhao, Hai  and
      Wang, Ping",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.911/",
    doi = "10.18653/v1/2025.emnlp-main.911",
    pages = "18040--18051",
    ISBN = "979-8-89176-332-6",
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.