import pandas as pd
from datasets import load_dataset, Dataset
import json 
import numpy as np 
import gc

def format_light_r1_sft_dataset(conversations, tokenizer, max_length): 
    dialogue = """<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. The assistant's final answer should be put within \\boxed{}.<|im_end|>\n"""
    for message in conversations: 
        if message['from'] == 'user':
            dialogue += f"<|im_start|>user\n{message['value']}<|im_end|>\n"
        elif message['from'] == 'assistant':
            answer = message['value']
            if "<think>" in answer and "</think>" in answer: 
                think_end = answer.index("</think>") + len("</think>")
                answer = answer[:think_end] + "\n<answer>" + answer[think_end:].strip() + "</answer>"
                dialogue += f"<|im_start|>assistant\n{answer}<|im_end|>\n"
    return tokenizer(dialogue, truncation=True, padding=False, max_length=max_length)

def format_aime_dataset(question, tokenizer): 
    dialogue = """<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. The assistant's final answer should be put within \\boxed{}.<|im_end|>\n"""
    dialogue += f"<|im_start|>user\n{question['problem']}<|im_end|>\n"
    dialogue += f"<|im_start|>assistant\n"
    answer = question['answer']
    tokenized_prompt = tokenizer(dialogue, padding=False, truncation=False)
    return {
        # 'prompt_text': dialogue, 
        'input_ids': tokenized_prompt['input_ids'],
        'attention_mask': tokenized_prompt['attention_mask'],
        'ground_truth_answer': answer 
    }

def load_dapo_dataset(tokenizer):
    dapo_ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k")

    train_df = dapo_ds['train'].to_pandas()

    cols_to_convert = []
    for col in train_df.columns:
        first_valid_index = train_df[col].first_valid_index()
        if first_valid_index is not None:
            first_item = train_df[col].loc[first_valid_index]
            if isinstance(first_item, (list, dict, np.ndarray)):
                try:
                    hash(first_item)
                except TypeError:
                    cols_to_convert.append(col)

    def to_sorted_json_string(item):
        if isinstance(item, (list, dict)):
            try:
                return json.dumps(item, sort_keys=True, separators=(',', ':'))
            except Exception:
                return str(item)
        elif isinstance(item, np.ndarray):
            try:
                return json.dumps(item.tolist(), sort_keys=True, separators=(',', ':'))
            except Exception:
                return str(item.tolist())
        return item

    converted_cols_info = {}
    for col in cols_to_convert:
        try:
            first_valid_index = train_df[col].first_valid_index()
            if first_valid_index is not None:
                converted_cols_info[col] = type(train_df[col].loc[first_valid_index])

            train_df[col] = train_df[col].apply(to_sorted_json_string)
        except Exception as e:
            if col in converted_cols_info:
                del converted_cols_info[col]

    unique_train_df = train_df.drop_duplicates()

    for col in converted_cols_info:
        if col in unique_train_df.columns:
                try:
                    unique_train_df[col] = unique_train_df[col].apply(
                        lambda x: json.loadapo_ds(x) if pd.notna(x) and isinstance(x, str) else x
                    )
                except Exception:
                    pass

    unique_train_dataset = Dataset.from_pandas(unique_train_df)
    dapo_ds['train'] = unique_train_dataset

    def format_for_training(example, tokenizer):
        prompt_text = json.loads(example['prompt'])[0]['content']
        ground_truth_text = json.loads(example['reward_model'])['ground_truth'] 
        tokenized_prompt = tokenizer(prompt_text, truncation=False, padding=False)
        return {
            'input_ids': tokenized_prompt['input_ids'],
            'attention_mask': tokenized_prompt['attention_mask'],
            'ground_truth': float(ground_truth_text)
        }

    formatted_dataset = dapo_ds.map(
        format_for_training,
        fn_kwargs={'tokenizer': tokenizer},
        batched=False, 
        remove_columns=dapo_ds.column_names['train'] 
    )

    del train_df
    del unique_train_df
    gc.collect()

    return formatted_dataset['train']

def load_light_r1_sft_dataset(tokenizer, context_length, num_proc):
    ds = load_dataset("qihoo360/Light-R1-SFTData")
    SFT_dataset = ds.map(
        lambda x: format_light_r1_sft_dataset(x["conversations"], tokenizer, max_length=context_length),
        remove_columns=['conversations'],
        num_proc=num_proc
    )
    return SFT_dataset['train']

def load_aime_dataset(tokenizer, version):
    if version == 'AIME_2024':
        AIME_ds = load_dataset("HuggingFaceH4/aime_2024")
    elif version == 'AIME_2025':
        AIME_ds = load_dataset("yentinglin/aime_2025", "default")

    formatted_AIME_ds = AIME_ds.map(
        format_aime_dataset, 
        fn_kwargs={'tokenizer': tokenizer},
        remove_columns=AIME_ds['train'].column_names
    )

    return formatted_AIME_ds['train']