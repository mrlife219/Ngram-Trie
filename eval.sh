MODEL = your_model_path

CUDA_VISIBLE_DEVICES=2 python hagrid/hagrid.py --model-path $MODEL
CUDA_VISIBLE_DEVICES=2 python sd-bench/sdbench.py --model-path $MODEL
CUDA_VISIBLE_DEVICES=2 python trivia/trivia.py --model-path $MODEL

