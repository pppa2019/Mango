import jsonlines
import argparse
import pprint
import sys
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, set_seed
from human_eval.data import write_jsonl, read_problems, stream_jsonl

# instruction_dict = {
#     "origin": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n{input}\n\n### Response: Here's the Python script for the given problem:""",
#     "cot": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem :\n{input}\n\n### Response: Here's the Python script for the given problem with explanation step by step via annotation:""",
#     "annotate2": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n{input}\n\n### Response: Here's the Python script for the given problem with annotation to explain the logic""",
#     "simple": """\n{input}\n"""
# }
instruction_dict = {
    "origin2": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n{input}\n\n### Response:""",
    "origin": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n{input}\n\n### Response: Here's the Python script for the given problem:""",
    "cot": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem :\n{input}\n\n### Response: Here's the Python script for the given problem with step by step reasoning first:""",
    "cot2": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem :\n{input}\n\n### Response: Make a coding plan step by step first, and the Python script for the given problem is as follows:""",
    "cot3": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem :\n{input}\n\n### Response: Think step by step first, then give the Python script directly:""",
    "annotate1": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n{input}\n\n### Response: Here's the Python script for the given problem with comments to explain the logic:""",
    "annotate2": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n{input}\n\n### Response: Here's the Python script for the given problem with annotation to explain the logic:""",
    "annotate3": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n{input}\n\n### Response: It would be better to annotation to explain the logic. Here's the Python script for the given problem:""",
    "annotate4": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n{input}\n\n### Response: Here's the Python script for the given problem (the difficult logic will be explained by simple comments):""",
    "comment_more": """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n{input}\n\n### Response: Here's the Python script for the given problem with detailed comments:""",
    "simple": """\n{input}\n"""
}

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def read_mbpp(path):
    mbpp_problems = {}
    with jsonlines.open(path, "r") as fin:
        for obj in fin:
            mbpp_problems[obj["task_id"]] = obj
    return mbpp_problems

def extract_text(prompt, remove_lines=True):
    token = '\"\"\"'
    start = token
    end = '>>>'

    start_idx = prompt.find(start) + len(start)
    end_idx = prompt.find(end)

    output = prompt[start_idx: end_idx]
    if remove_lines:
        output = output.replace('\n', ' ')
    output = re.sub(r"\s+", " ", output).strip()

    return output

# def generate_prompt(input):
#     INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:
# Create a Python script for this problem:
# {input}

# ### Response:"""
#     return INSTRUCTION

def generate_prompt(input, ins_type="origin"):
    INSTRUCTION = instruction_dict[ins_type].format(input=input)
    return INSTRUCTION

def get_model(
    load_8bit: bool = False,
    base_model: str = "bigcode/starcoder",
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
    )
    if 'codellama' in base_model.lower():
        from transformers import LlamaForCausalLM, CodeLlamaTokenizer
        model_type =  LlamaForCausalLM
        tokenizer_type = CodeLlamaTokenizer
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float16
        tokenizer_type =  AutoTokenizer
        model_type = AutoModelForCausalLM
    tokenizer = tokenizer_type.from_pretrained(base_model)
    print('device',device)
    if device == "cuda":
        model = model_type.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=model_dtype,
            device_map="auto",
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    model.config.pad_token_id = tokenizer.pad_token_id

    if not load_8bit:
        model.bfloat16()  # seems to fix bugs for some users.

    model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--prompt_type', type=str, default=None, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=512, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    parser.add_argument('--overwrite', action='store_true', help='')
    parser.add_argument('--mbpp_path', type=str, help="")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_test', action='store_true')


    args = parser.parse_args()

    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    set_seed(args.seed)
    prompt_type = args.prompt_type

    STOP_SEQS = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']

    problems = read_mbpp(args.mbpp_path)

    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = []
    if 'text' in problems[task_ids[0]]:
        read_key = 'text'
    else:
        read_key = 'prompt'
    for task_id in task_ids:
        if args.skip_test:
            prompt = f"\n{problems[task_id][read_key]}\n"
        else:
            prompt = f"\n{problems[task_id][read_key]}\nTest examples:"
            if task_id == 493:
                # The test examples are too long. We choose to only include the function name.
                test_example = problems[task_id]['test_list'][0]
                prompt += f"\ncalculate_polygons(startx, starty, endx, endy, radius)"
            else:
                for test_example in problems[task_id]['test_list']:
                    prompt += f"\n{test_example}"
        prompts.append(prompt)
    
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))
    do_sample = False
    if args.temperature > 0.0001:
        do_sample = True

    tokenizer, model = get_model(base_model=args.model)
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        do_sample=do_sample,
        temperature=args.temperature,
        max_length=args.max_len,
        num_return_sequences=args.num_seqs_per_iter,
        eos_token_id=tokenizer.eos_token_id,
        top_p=0.95
    )

    print(f"Loaded {args.model}.")
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_prompt(prompt, prompt_type)]

        ids_batch = [task_ids[i]]

        completion_seqs = []

        encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=args.max_len, return_token_type_ids=False).to(device)

        if args.decoding_style == 'sampling':
            loops = int(args.N / args.num_seqs_per_iter)
        else:
            loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                if args.decoding_style == 'sampling':
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
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

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