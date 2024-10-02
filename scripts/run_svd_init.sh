#
#1.min or max
#2.rank
#3.basemodel_path
#4.save_root_path
#5.hyperparameter, e.g., llm-adapters

mkdir -p ./svd_init_models

CUDA_VISIBLE_DEVICES=2 python svd_init.py "min" 64 "../../models/llama-2-7b" "./svd_init_models" "LLM-Adapters" &
# CUDA_VISIBLE_DEVICES=3 python svd_init.py "min" 64 "../../models/llama-2-7b" "./svd_init_models" "QLoRA" &

# can do parallel inits with different rank and min/max