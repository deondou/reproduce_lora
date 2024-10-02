#    Copyright 2024 Fanxu Meng, Zhaohui Wang, Muhan Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Literal

import torch
import transformers
from transformers import Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel, AdaLoraConfig, TaskType

IGNORE_INDEX = -100
PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    adapter_name_or_path: Optional[str] = field(default=None)
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_split: str = field(
        default="train[:100000]", metadata={"help": "(`['train', 'test', 'eval']`):"}
    )
    dataset_field: List[str] = field(
        default=None, metadata={"help": "Fields of dataset input and output."}
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    lora_r: int = field(default=None, metadata={"help": "The rank of the adapter. When passing `None` and `adapter_name_or_path` is also `None`, full fine-tuning is used."})
    lora_alpha: int = field(default=None, metadata={"help": "The rank of the adapter. When passing `None` and `adapter_name_or_path` is also `None`, full fine-tuning is used."})
    lora_dropout: float = field(default=0.05, metadata={"help": "The dropout of the adapter. When passing `None` and `adapter_name_or_path` is also `None`, full fine-tuning is used."})
    target_modules: List[str] = field(
        default_factory=lambda:["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"], metadata={"help": "target modules name"}
    )
    init_lora_weights: Literal[True, "pissa"] = field(
        default=True,
        metadata={
            "help": (
                "Passing True (default) results in the LoRA initialization." 
                "Passing `pissa` results in PiSSA initialization."
            ),
        },
    )
    method_type: str = field(default="lora", metadata={"help": "The method type of the adapter."})
    use_dora: bool = field(default=False, metadata={"help": "Whether to use Dora."})
    use_rslora: bool = field(default=False, metadata={"help": "Whether to use RSLora."})
    use_adalora: bool = field(default=False, metadata={"help": "Whether to use AdaLora."})
    loraplus_lr_ratio: int = field(default=4, metadata={"help": "The ratio of the learning rate of the LoRA+."})
    


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    if script_args.target_modules:
        print(f"script_args.target_modules {script_args.target_modules}")
        target_modules = script_args.target_modules[0].split(',')
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
    if "lora+" in script_args.method_type or "loraplus" in script_args.method_type:
        method_type = script_args.method_type.split("-")[0]
        script_args.loraplus_lr_ratio = int(script_args.method_type.split("-")[1])
        script_args.method_type = method_type
    if script_args.method_type == "lora" or script_args.method_type == "pissa" or script_args.method_type == "milora" or script_args.method_type == "lora+" or script_args.method_type == "loraplus":
        script_args.use_rslora = False
        script_args.use_dora = False
        script_args.use_adalora = False
    elif script_args.method_type == "rslora":
        script_args.use_rslora = True
        script_args.use_dora = False
        script_args.use_adalora = False
    elif script_args.method_type == "dora":
        script_args.use_rslora = False
        script_args.use_dora = True
        script_args.use_adalora = False
    elif script_args.method_type == "adalora":
        script_args.use_rslora = False
        script_args.use_dora = False
        script_args.use_adalora = True
    print(script_args)    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
    )
    if "min" in script_args.model_name_or_path or "max" in script_args.model_name_or_path:
        script_args.adapter_name_or_path = script_args.model_name_or_path + "/lora"
    print(f"script_args.model_name_or_path {script_args.model_name_or_path}")
    print(f"script_args.adapter_name_or_path {script_args.adapter_name_or_path}")
    if script_args.adapter_name_or_path is not None:
        print(f"Load {script_args.init_lora_weights} from {script_args.adapter_name_or_path}: ",)
        model = PeftModel.from_pretrained(model, script_args.model_name_or_path, subfolder="./lora", is_trainable=True)
    elif script_args.lora_r is not None:
        print(f"Initilized {script_args.init_lora_weights} layers")
        if script_args.use_adalora:
            lora_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            target_modules=target_modules,
            total_step=int(24688),
            lora_dropout=script_args.lora_dropout,
            deltaT = 100,
            init_r = int(script_args.lora_r*1.5),
            tfinal = 100,
            )
        else:
            if script_args.use_rslora + script_args.use_dora > 1:
                raise ValueError("At most one of use_rslora, use_dora can be True.")
            lora_config = LoraConfig(
                r=script_args.lora_r,
                lora_alpha=script_args.lora_r if script_args.lora_alpha is None else script_args.lora_alpha,
                init_lora_weights = script_args.init_lora_weights,
                target_modules = target_modules,
                lora_dropout=script_args.lora_dropout,
                use_rslora=script_args.use_rslora,
                use_dora=script_args.use_dora,
                bias="none",
                task_type="CAUSAL_LM",
            )
            print(lora_config)
        
        model = get_peft_model(model, lora_config)
    else:
        print("Full Parameter Fine-Tuning")
                
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: requires_grad={param.requires_grad}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    raw_train_datasets = load_dataset(script_args.data_path, split=script_args.dataset_split)

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": script_args.dataset_field[0], "response": script_args.dataset_field[1]}
    )

    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)

    if script_args.method_type == "lora+" or script_args.method_type == "loraplus":
        from peft.optimizers import create_loraplus_optimizer
        from torch.optim import AdamW
        ## TODO: verify the correctness of the optimizer cls, utlization of the lr, and add arguments for the loraplus_lr_ratio
        optimizer = create_loraplus_optimizer(
            model = model,
            optimizer_cls = AdamW,
            lr = script_args.learning_rate,
            loraplus_lr_ratio=script_args.loraplus_lr_ratio,
            )

        # scheduler = use constant scheduler in the paper
        class CustomTrainer(Trainer):
            def create_optimizer_and_scheduler(self, num_training_steps: int):
                self.optimizer = optimizer  # 

        trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)


    else:
        trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
    model.config.use_cache = False
    trainer.train()
    trainer.save_state()
    model.save_pretrained(os.path.join(script_args.output_dir,'ft'))

if __name__ == "__main__":
    train()
