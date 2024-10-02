import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
import peft
from peft import LoraConfig, TaskType, get_peft_model
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_seed(42)

total_time=0

def initialize_lora_layer(weights, rank, mode="min"):
    start = time.time()
    U, S, V = torch.linalg.svd(weights, full_matrices=False)
    end = time.time()
    delta_time = end - start
    logger.info(f"delta_time for linglg.svd{delta_time}")
    global total_time
    total_time += end - start
    lora_alpha=rank
    # import pdb
    # pdb.set_trace()
    if mode == "min":
        U_select = U[:, -rank:]
        S_select = S[-rank:]
        V_select = V[-rank:, :]
    elif mode == "mid":
        mid_start = (len(S) - rank) // 2
        mid_end = mid_start + rank
        U_select = U[:, mid_start:mid_end]
        S_select = S[mid_start:mid_end]
        V_select = V[mid_start:mid_end, :]
    elif mode == "max":
        U_select = U[:, :rank]
        S_select = S[:rank]
        V_select = V[:rank, :]
    elif mode == "random":
        indices = np.random.choice(len(S), rank, replace=False)
        indices = np.sort(indices)
        U_select = U[:, indices]
        S_select = S[indices]
        V_select = V[indices, :]
    else:
        raise ValueError("Unknown mode!")

    scaling = lora_alpha / rank
    S_select /= scaling  # 这里增加scaling后，可以允许lora_alpha和lora_rank不一致
    S_sqrt = torch.sqrt(S_select)
    B = U_select @ torch.diag(S_sqrt)
    A = torch.diag(S_sqrt) @ V_select
    delta = scaling * B @ A

    return A, B, delta


def move_lora_file(SAVE_PATH):
    import os
    import shutil
    target_path = SAVE_PATH
    lora_path = os.path.join(target_path, 'lora')
    os.makedirs(lora_path, exist_ok=True)

    # 移动文件到 'lora' 目录
    files_to_move = ['adapter_config.json', 'adapter_model.bin']
    for file_name in files_to_move:
        src_file = os.path.join(target_path, file_name)
        dst_file = os.path.join(lora_path, file_name)
        if os.path.exists(src_file):
            shutil.move(src_file, dst_file)
            print(f"Moved {src_file} to {dst_file}")
        else:
            print(f"{src_file} does not exist")

    print("Files moved successfully.")
    return

def svd_tailor_and_save(args):
    mode=args.mode
    svd_rank=args.svd_rank
    MODEL_PATH=args.model_path
    SAVE_PATH=args.save_path
    hyper_param_type=args.hyper_param_type

    SAVE_PATH+=f"/{hyper_param_type}-rank-{svd_rank}"
    if mode == "min":
        SAVE_PATH += "-min"
    elif mode == "mid":
        SAVE_PATH += "-mid"
    elif mode == "max":
        SAVE_PATH += "-max"
    elif mode == "random":
        SAVE_PATH += "-random"
    else:
        print("NOT LEGAL MODE")

    # svd_rank=64
    if hyper_param_type == "LLM-Adapters":
        lora_rank = svd_rank
        lora_alpha = svd_rank
        lora_dropout = 0.05
        target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
    elif hyper_param_type == "QLoRA":
        lora_rank = svd_rank
        lora_alpha = svd_rank
        lora_dropout = 0.1
        target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']

    else:
        raise ValueError(f"Unknown hyper_param_type{hyper_param_type}")
    # lora_rank = svd_rank
    # lora_alpha = svd_rank
    # lora_dropout = 0.05
    # target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
    # target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
    # target_modules = ["q_proj", "v_proj"]
    device = "cuda:0"
    # device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    peft_config = LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                r=lora_rank,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
            )
    model = get_peft_model(model, peft_config).to(device)

    logger.info(f"Start processing SVD and Lora initialization...")
    import time
    start = time.time()
    last_time = start
    with torch.no_grad():
        for n, p in model.named_parameters():
            if any(proj in n for proj in target_modules) and "lora" not in n:
                parent_name = n.split(".base_layer.weight")[0]
                parent_module = model.get_submodule(parent_name)
                lora_A_init, lora_B_init, delta = initialize_lora_layer(p.data.float(), lora_rank, mode=mode)
                # lora_A_init, lora_B_init, delta = stable_initialize_lora_layer(p.data.float(), lora_rank, mode=mode)

                parent_module.base_layer.weight.data -= delta
                parent_module.lora_A['default'].weight.data = lora_A_init
                parent_module.lora_B['default'].weight.data = lora_B_init
                current = time.time()
                logger.info(f"processed: {parent_name} mode:{mode} svd_rank:{svd_rank} time cost:{current-last_time}")
                last_time = current
                # logger.info(f"processed: {parent_name} mode:{mode} svd_rank:{svd_rank}")
    end = time.time()
    logger.info(f"Total time cost for SVD: {end-start}")
    global total_time
    logger.info(f"Specific time cost for SVD: {total_time}")

    logger.info(f"Save SVD model and initial lora to {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH+"/lora", safe_serialization=False)
    base_model = model.unload()
    base_model.save_pretrained(SAVE_PATH, safe_serialization=False)
    tokenizer.save_pretrained(SAVE_PATH)
    # move_lora_file(SAVE_PATH)
    logger.info(f"Finished!")



import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="argparse for parallel svd tailor")

# 添加参数
parser.add_argument("mode", type=str, help="svd mode", default="min")
parser.add_argument("svd_rank", type=int, help="An integer number",default=64)
parser.add_argument("model_path",type=str, help="base model path for svd tailor", default="/data/xzg/models/meta-llama/Llama-2-7b-hf")
parser.add_argument("save_path",type=str, help="save model path for svd tailor", default="/data/whq/projects/svd_lora/svd_init_models")
parser.add_argument("hyper_param_type",type=str, help="hyper param type", default="llm-adapters")
# 解析命令行输入
args = parser.parse_args()




svd_tailor_and_save(args=args)
