# define your own save directory
root=$1
mkdir -p $root
math_root=$root/math
code_root=$root/code
mkdir -p $math_root
mkdir -p $code_root

mkdir -p results
mkdir -p results/gsm8k
mkdir -p results/MATH
mkdir -p ./logs

math_merge_and_infer(){
    BASE_MODEL=$1
    OUTPUT_name=$2
    GPU_ID=$3
    OUTPUT=$math_root/$OUTPUT_name
    adapter_path=$OUTPUT/ft
    output_path=${adapter_path}-merged
    CUDA_VISIBLE_DEVICES=$GPU_ID python merge_adapter_to_base_model.py --base_model $BASE_MODEL --adapter $adapter_path --output_path $output_path
    CUDA_VISIBLE_DEVICES=$GPU_ID python inference/gsm8k_inference.py --model $output_path &> results/gsm8k/gsm8k-${OUTPUT_name}.log
    CUDA_VISIBLE_DEVICES=$GPU_ID python inference/MATH_inference.py --model $output_path &> results/MATH/MATH-${OUTPUT_name}.log

}

train() {
    BASE_MODEL=$1
    EPOCHS=$2
    RANK=$3
    GPUS=$4
    MASTER_PORT=$5
    DATA=$6
    SETTING=$7
    METHOD=$8
    
    
    if [ "$DATA" = "meta-math/MetaMathQA" ]; then
        field_2=response
        root=$math_root
        TRAIN_TASK=math
        per_device_train_batch_size=1
    fi

    num_GPUs=$(echo $GPUS | tr ',' '\n' | wc -l)

    if [ "$SETTING" = "LLM-Adapters" ]; then
        TOTAL_BATCH_SIZE=16
        LR=3e-4
        WARM_UP_RATIO=0
        WARM_UP_STEPS=100
        LR_SCHEDULER="linear" 
        LORA_ALPHA=$(($RANK * 2))
        MAX_LEGNTH=2048
        DROPOUT=0.05
        TARGET="q_proj,k_proj,v_proj,up_proj,down_proj"
    fi

    
    gradient_accumulation_steps=$((TOTAL_BATCH_SIZE / (per_device_train_batch_size * num_GPUs)))

    OUTPUT_name=${SETTING}-${METHOD}-LR-${LR}-${DATA_NUM}-EPOCHS-${EPOCHS}-rank-${RANK}

    LOGNAME=logs/${TRAIN_TASK}-${OUTPUT_name}.log

    if [[ "$OUTPUT_name" == *"pissa"* ]] || [[ "$OUTPUT_name" == *"milora"* ]]; then
        LORA_ALPHA=$(($RANK))
    fi
    
    
    echo "LORA_ALPHA: $LORA_ALPHA"
    
    OUTPUT=${root}/$OUTPUT_name

    deepspeed --master_port $MASTER_PORT --include localhost:$GPUS train.py \
        --deepspeed configs/stage2.conf \
        --model_name_or_path $BASE_MODEL \
        --output_dir $OUTPUT \
        --lora_r $RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $DROPOUT \
        --target_modules $TARGET \
        --data_path $DATA \
        --dataset_split "train[:100000]" \
        --dataset_field query $field_2 \
        --model_max_length $MAX_LEGNTH \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --save_strategy "epoch" \
        --learning_rate $LR \
        --weight_decay 0. \
        --warmup_ratio $WARM_UP_RATIO \
        --warmup_steps $WARM_UP_STEPS \
        --lr_scheduler_type $LR_SCHEDULER \
        --logging_steps 1 \
        --bf16 True \
        --tf32 True \
        --method_type $METHOD \
        --report_to tensorboard &> $LOGNAME
    
    if [ "$TRAIN_TASK" = "math" ]; then
        device=$(echo $GPUS | cut -d',' -f1)
        math_merge_and_infer $BASE_MODEL $OUTPUT_name $device
    fi

}


# BASE_MODEL=$1 
# EPOCHS=$2 
# RANK=$3
# GPUS=$4 
# lr=$5 
# MASTER_PORT=$6 
# DATA=$7 
# SETTING=$8 
# METHOD=$9

train /root/autodl-tmp/meta-llama/Llama-2-7b-hf 3 64 "0" 29500 meta-math/MetaMathQA LLM-Adapters lora 


