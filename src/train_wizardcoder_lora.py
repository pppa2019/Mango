#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import torch
import torch.distributed
import transformers
from torch.utils.data import Dataset
# from transformers import Trainer
from datasets import load_dataset
import os
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers.trainer_callback import TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl, AutoConfig, set_seed, Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from my_modeling_gpt_bigcode import My_GPTBigCodeForCausalLM
from my_modeling_llama import My_LlamaForCausalLM
# import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_BOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<|endoftext|>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_reverse": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction: Given a Python code, please predict its problem description: \n{instruction}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_input_comment": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: Here's the Python script for the given problem with annotation to explain the logic"
    ),
    "prompt_no_input_comment": (
        "Below is an instruction that describes a task, paired with an input that provides further context."
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Here's the Python script for the given problem with annotation to explain the logic"
    ),
    
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="bigcode/starcoder")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    lora_config: Optional[str] = field(
        default=None,
        metadata={"help": "LoRA config file."},
    )
    freeze_emb: Optional[bool] = field(
        default=True
    )
    contrastive_train:  Optional[bool] = field(
        default=False
    )
    train_target: Optional[str] = field(
        default="origin"
    )

    margin: Optional[float] = field(
        default=None
    )



class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


# xxx: save peft at train end
class SavePeftModelAtEndCallback(TrainerCallback):
    def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        peft_model_path = os.path.join(args.output_dir, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(state.best_model_checkpoint, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

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
    dual: bool=False,
    reverse_sources: Sequence[str]=None,
    reverse_targets: Sequence[str]=None,
    set_reward_score: bool=False,
    mask_comment: bool=False,
    neg_samples: Sequence[str]=None
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    if neg_samples:
        pure_code_tokenized = _tokenize_fn(neg_samples, tokenizer)
        pure_code_ids = pure_code_tokenized['input_ids']
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    if mask_comment:
        for target, label, source_len in zip(targets, labels, sources_tokenized["input_ids_lens"]):
            all_lines = target.split('\n')
            all_comments = [line.strip().split('#')[-1] if len(line.strip())>0 and '#' in line.strip() else '' for line in target.split('\n') ]
            for line_i in range(len(all_lines)):
                end_idx = len(tokenizer.tokenize('\n'.join(all_lines[:line_i+1]))) + source_len
                if all_comments[line_i].strip() == '':
                    continue
                begin_idx = end_idx - len(tokenizer.tokenize('#'+ all_comments[line_i]))
                label[begin_idx:end_idx] = IGNORE_INDEX
    if dual:
        reverse_examples = [s + t for s, t in zip(reverse_sources, reverse_targets)]
        reverse_examples_tokenized, reverse_sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (reverse_examples, reverse_sources)]
        reverse_input_ids = reverse_examples_tokenized["input_ids"]
        reverse_labels = copy.deepcopy(reverse_input_ids)
        for label, source_len in zip(reverse_labels, reverse_sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels, reverse_input_ids=reverse_input_ids, reverse_labels=reverse_labels)
    if set_reward_score:
        # calculate reward score via inline comment ratio
        reward_score_list = []
        for target in targets:
            all_comments = [line.strip().split('#')[-1] for line in target.split('\n') if len(line.strip())>0 and '#' in line.strip()]
            comment_count = len(all_comments)
            if comment_count==0:
                reward_score_list.append(0)
            elif comment_count>=3 and comment_count<=7:
                reward_score_list.append(1)
            else:
                reward_score_list.append(0)
        return dict(input_ids=input_ids, labels=labels, reward_score=reward_score_list)
    if pure_code_ids is not None:
        return dict(input_ids=input_ids, labels=labels, pure_code_ids=pure_code_ids)
    return dict(input_ids=input_ids, labels=labels)



def comment_preprocess(
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
    # abstract comment from output
    pure_code_list = []
    def clean_comment(origin_code):
        result = []
        for line in origin_code.split('\n'):
            if  '#' in line:
                result.append(line.split('#')[0])
            else:
                result.append(line)
        if len(result)==0:
            return 'None'
        return '\n'.join(result)
    for target in targets:
        pure_code_list.append(clean_comment(target))
    pure_code_tokenized = _tokenize_fn(pure_code_list, tokenizer)
    pure_code_ids = pure_code_tokenized["input_ids"]
    return dict(input_ids=input_ids, labels=labels, pure_code_ids=pure_code_ids)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # import ipdb;ipdb.set_trace()
        assert 'pure_code_ids' in instances[0], "Error! pure_code_ids"
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if 'pure_code_ids' in instances[0]:
            pure_code_ids = [instance['pure_code_ids'] for instance in instances]
            pure_code_ids = [torch.tensor(x) for x in pure_code_ids]
            pure_code_ids = torch.nn.utils.rnn.pad_sequence(
                pure_code_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            return dict(
                input_ids=input_ids,
                labels=labels,
                pure_code_ids=pure_code_ids,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
        return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )

def train_tokenize_function(examples, tokenizer, set_reward_score=False, mask_comment=False):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    if 'input' in examples:
        sources = [
            prompt_input.format_map(dict(instruction=instruction, input=input)) if input != "" \
            else prompt_no_input.format_map(dict(instruction=instruction)) \
            for instruction, input in zip(examples['instruction'], examples['input']) 
        ]
    else:
        sources = [
            prompt_no_input.format_map(dict(instruction=instruction)) \
            for instruction in examples['instruction']
        ]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer, set_reward_score=set_reward_score, mask_comment=mask_comment)
    return data_dict


# rewrite tokenize for comment contrastive samples
def train_comment_tokenize_function(examples, tokenizer):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    if 'input' in examples:
        sources = [
            prompt_input.format_map(dict(instruction=instruction, input=input)) if input != "" \
            else prompt_no_input.format_map(dict(instruction=instruction)) \
            for instruction, input in zip(examples['instruction'], examples['input']) 
        ]
    else:
        sources = [
            prompt_no_input.format_map(dict(instruction=instruction)) \
            for instruction in examples['instruction']
        ]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples['output']]
    if 'negative' not in examples:
        data_dict = comment_preprocess(sources, targets, tokenizer)
    else:
        data_dict = preprocess(sources, targets, tokenizer, neg_samples=[neg_code for neg_code in examples['negative']])
    return data_dict


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    tokenizer_type = transformers.AutoTokenizer
    tokenizer = tokenizer_type.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.ins_token_id =  tokenizer.convert_tokens_to_ids(['###'])[0]
    config.code_token_id =  tokenizer.convert_tokens_to_ids(['```'])[0]

    if training_args.train_target:
        config.train_target = training_args.train_target
    if training_args.margin is not None:
        config.margin = training_args.margin
    config.contrastive_train = training_args.contrastive_train
    if 'bigcode' in config.architectures[0].lower():
        model = My_GPTBigCodeForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            config=config
        )
    else:
        model = My_LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            config=config
        )
    if training_args.lora_config:
        lora_hyper = json.load(open(training_args.lora_config))
        for key, value in lora_hyper.items():
            print("{} : {}".format(key, value))
        lora_config = LoraConfig(
            r=lora_hyper['lora_r'],
            lora_alpha=lora_hyper['lora_alpha'],
            target_modules=lora_hyper['lora_target_modules'],
            lora_dropout=lora_hyper['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
            # enable_lora=lora_hyper['enable_lora']
        )
        model = get_peft_model(model, lora_config)
        print(f"LoRA configs: {lora_config}")
    # xxx: To avoid "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
    # xxx: Seems due to gradient_checkpointing, to check later
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if training_args.freeze_emb:
        for name, param in model.named_parameters():
            if 'embed' in name or 'wte' in name:
                print("Freeze parameter:", name)
                param.requires_grad = False
            
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "starcoder" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )

    raw_train_datasets = load_dataset('json', data_files=data_args.data_path, split="train", cache_dir=training_args.cache_dir)
    if training_args.local_rank > 0: 
        torch.distributed.barrier()
    if training_args.contrastive_train:
        tokenize_fn = train_comment_tokenize_function
    else:
        tokenize_fn = train_tokenize_function
    
    if not (training_args.contrastive_train or training_args.contrastive_prompt_train) :
        train_dataset = raw_train_datasets.map(
            tokenize_fn,
            batched=True,
            batch_size=256,
            num_proc=32,
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=False, # not args.overwrite_cache
            desc="Running tokenizer on train dataset",
            fn_kwargs={"tokenizer": tokenizer, "set_reward_score": training_args.set_reward_score, 'mask_comment':training_args.mask_comment}
        ).shuffle()
     
    if training_args.contrastive_train:
        train_dataset = raw_train_datasets.map(
            tokenize_fn,
            batched=True,
            batch_size=256,
            num_proc=32,
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=False, # not args.overwrite_cache
            desc="Running tokenizer on train dataset",
            fn_kwargs={"tokenizer": tokenizer}
        ).shuffle()  


    if training_args.local_rank == 0 and torch.cuda.device_count() > 1:
        torch.distributed.barrier()
        #Tell Trainer not to attempt DataParallel
        model.is_parallelizable = True
        model.model_parallel = True
    if training_args.local_rank == 0:
        print(len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    
    if training_args.lora_config:
        callbacks = [SavePeftModelCallback, SavePeftModelAtEndCallback]
    else:
        callbacks = None
    
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        callbacks=callbacks,
        **data_module)
 
    model.config.use_cache = False
    if training_args.lora_config:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

    trainer.train()
    trainer.save_state()
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()