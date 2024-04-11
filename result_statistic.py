import json
import os
from collections import Counter
import json
import numpy as np
from random import sample
import argparse

def count_annotation(data):
    code_blocks = []
    annotation_blocks = []
    code_rates = []
    for item in data:
        code_item = item['completion']
        code_lines = [line for line in code_item.split('\n') if line!='']
        annotate_lines = [line.strip() for line in code_item.split('\n') if len(line.strip())>0 and line.strip()[0]=='#']
        code_blocks.append(code_lines)
        code_rates.append(len(item['completion'].split('\n')))
        annotation_blocks.append(annotate_lines)
    code_line_count = [len(lines) for lines in code_blocks]
    annotation_line_count = [len(lines) for lines in annotation_blocks]
    return np.mean(code_line_count), np.mean(annotation_line_count), np.mean(code_rates)

def get_sample_python_data(data, sample_id=None):
    if sample_id is None:
        sample_data = sample(data, 5)
    else:
        sample_data = []
        for i in sample_id:
            sample_data.append(data[i])
    sample_output = [item['output'] for item in sample_data]
    return sample_data, sample_output

def parse_python_blocks(input_string):
    segs = input_string.split('```python')
    segs = segs[1:]
    code_blocks = []
    for seg in segs:
        code_blocks.append(seg.split('```')[0])
    return code_blocks[0:1]

def load_jsonl(file):
    data = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data

def show_counter(data):
    result = [item['result'] for item in data]
    counter = Counter(result)
    test_not_pass_count = 0
    for key, value in counter.items():
        if 'failed: test' in key.lower():
            test_not_pass_count += value
    return counter

def apart_pf_cases(data):
    # pass_cases = []
    fail_dict = {f"HumanEval/{i}":[] for i in range(164)}
    for item in data:
        if 'passed' in item['result'].lower():
            if item['task_id'] in fail_dict:
                fail_dict.pop(item['task_id'])
        else:
            if item['task_id'] in fail_dict:
                fail_dict[item['task_id']].append(item)
            # fail_cases.append(item)
    fail_cases_passM = []
    for value in fail_dict.values():
        fail_cases_passM.extend(value)
    return fail_cases_passM

def parse_commet(code_string):
    code_lines = [line for line in code_string.split('\n') if line!='']
    annotate_lines = [line.strip() for line in code_lines if len(line.strip())>0 and line.strip()[0]=='#']
    pure_code_lines = [line.strip() for line in code_lines if len(line.strip())>0 and line.strip()[0]!='#']
    return '\n'.join(pure_code_lines), '\n'.join(annotate_lines)


# parse the data to independent parts, including problem description, pure code and comment lines.
def parse_data2elements(data):
    descriptions = []
    pure_codes = []
    comment_lines = []
    extra_outputs = []
    for item in data:
        descriptions.append(item['all_code'].split('### Instruction:\n')[-1].split('\n\n### Response:')[0])
        des_snap = '\n'.join(item['all_code'].split('### Instruction:\n')[-1].split('\n\n### Response:')[0].split('\n')[-4:])
        gen_code_string = item['completion'].split(des_snap)[-1]
        pure_code, comment = parse_commet(gen_code_string)
        pure_codes.append(pure_code)
        comment_lines.append(comment)
        extra_outputs.append(item['all_code'].split('\n\n### Response:')[-1].split('```')[-1])
    
    return descriptions, pure_codes, comment_lines, extra_outputs
    





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    file = args.path
    data = load_jsonl(file)


    # get passed and failed data
    # pass_cases, 
    fail_cases = apart_pf_cases(data)
    fail_counter = show_counter(fail_cases)
    verified_fail = 0
    first_test_fail = 0
    other_fail = 0
    # fail_counter.pop('passed')
    for key, value in fail_counter.items():
        # if 'Test' in key or 'assert fails' in key or 'test error' in key:
        if 'Test 1' in key or 'assert fails 1' in key:
            first_test_fail += value
        elif 'Test ' in key or 'assert fails' in key:
            other_fail += value
        else:
            verified_fail += value
    print('verified fail', verified_fail)
    print('first test fail', first_test_fail)
    print('other fail', other_fail)
    total_len = len(data)
    print("Error Type Distribution", round(verified_fail/total_len, 4), round(first_test_fail/total_len, 4), round(other_fail/total_len, 4))

    # output style statistic
    style_result = count_annotation(data)
    print(style_result)
    