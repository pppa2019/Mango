import argparse
import pprint
import sys
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig, set_seed
from human_eval.data import write_jsonl, read_problems, stream_jsonl
from peft import PeftModel,set_peft_model_state_dict
from my_modeling_gpt_bigcode import My_GPTBigCodeForCausalLM
from my_modeling_llama import My_LlamaForCausalLM
from transformers.models.llama import LlamaForCausalLM
from transformers.models.gpt_bigcode import GPTBigCodeForCausalLM
import json
    
prefix_ins = """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n{input}\n\n### Response: """

instruction_dict = {
    "origin2": prefix_ins,
    "origin": prefix_ins + """Here's the Python script for the given problem:""",
    "cot": prefix_ins +"""Here's the Python script for the given problem with step by step reasoning first:""",
    "cot2": prefix_ins +"""Make a coding plan step by step first, and the Python script for the given problem is as follows:""",
    "cot3": prefix_ins +"""Think step by step first, then give the Python script directly:""",
    "annotate1": prefix_ins + """Here's the Python script for the given problem with comments to explain the logic:""",
    "annotate2": prefix_ins + """Here's the Python script for the given problem with annotation to explain the logic:""",
    "annotate3":  prefix_ins +"""It would be better to annotation to explain the logic. Here's the Python script for the given problem:""",
    "annotate4":  prefix_ins +"""Here's the Python script for the given problem (the difficult logic will be explained by simple comments):""",
    "comment_more":  prefix_ins +"""Here's the Python script for the given problem with detailed comments:""",
    "simple": """\n{input}\n"""
}

ext_prompt_dict = json.load(open('prompt_ext.json'))
for key, value in ext_prompt_dict.items():
    instruction_dict[key] = prefix_ins + value


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def generate_prompt(input, ins_type="origin"):
    INSTRUCTION = instruction_dict[ins_type].format(input=input)
    return INSTRUCTION

def get_model(
    load_8bit: bool = False,
    base_model: str = "bigcode/starcoder",
    lora_weights: str = None
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
    )
    config = AutoConfig.from_pretrained(base_model)
    print(config._name_or_path)
    model_dtype = torch.float16
    tokenizer_type =  AutoTokenizer
    model_type = AutoModelForCausalLM
    tokenizer = tokenizer_type.from_pretrained(base_model)
    print(device)
    if device == "cuda":        
        model = model_type.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=model_dtype,
            # device_map="auto",
        ).cuda()
    elif device == "mps":
        model = eval(config.architectures[0]).from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=model_dtype,
        ).cuda()
    if lora_weights:
        print('Using Lora:', lora_weights)
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16
        )
    model.config.pad_token_id = tokenizer.pad_token_id

    model.bfloat16()

    model.eval()
    
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
    parser.add_argument('--lora_weights', type=str, default=None, help="")
    parser.add_argument('--prompt_type', type=str, default=None, help="")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=512, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    parser.add_argument('--greedy_decode', action='store_true', help='')
    parser.add_argument('--overwrite', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    set_seed(args.seed)
    prompt_type = args.prompt_type
    problems = read_problems()
    if args.temperature <1e-5:
        args.greedy_decode = True

    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))
    tokenizer, model = get_model(base_model=args.model, lora_weights=args.lora_weights)
    
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False if args.greedy_decode else True,
        temperature=args.temperature,
        max_length=args.max_len,
        num_return_sequences=args.num_seqs_per_iter,
        eos_token_id=tokenizer.eos_token_id,
        top_p=0.95
    )

    print(f"Loaded {args.model}.")
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        if os.path.exists(output_file):
            print(f'Skip {output_file} as it already exists')
            continue

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_prompt(prompt, prompt_type)]

        ids_batch = [task_ids[i]]

        completion_seqs = []
        if 'phi' in args.model:
            return_attention_mask = False
        else:
            return_attention_mask = True
        encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=args.max_len, return_token_type_ids=False, return_attention_mask=return_attention_mask).to(device)

        if args.decoding_style == 'sampling':
            loops = int(args.N / args.num_seqs_per_iter)
        else:
            loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                gen_tokens = model.generate(
                    **encoding,
                    generation_config=generation_config
                )

            if gen_tokens is not None:
                gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            else:
                gen_seqs = None

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    if "### " in gen_seq:
                        completion_seq = gen_seq.split("### Response:")[1]
                    else:
                        completion_seq = gen_seq
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')
                    # print(completion_seq)
                    completion_seqs.append(
                        {'task_id': task_id,
                         'completion': completion_seq,
                         'all_code': all_code,
                         }
                    )

        print("Saving results to {}".format(output_file))
        write_jsonl(output_file, completion_seqs)


if __name__ == '__main__':
    main()