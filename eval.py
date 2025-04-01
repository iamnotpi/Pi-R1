from datasets import load_dataset
import regex as re 
import torch 
from typing import Optional


def format_aime(question, tokenizer): 
    dialogue = """<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. The assistant's final answer should be put within \\boxed{}.<|im_end|>\n"""
    dialogue += f"<|im_start|>user\n{question['problem']}<|im_end|>\n"
    dialogue += f"<|im_start|>assistant\n"
    answer = question['answer']
    tokenized_prompt = tokenizer(dialogue, padding=False)
    return {
        'prompt_text': dialogue, 
        'input_ids': tokenized_prompt['input_ids'],
        'attention_mask': tokenized_prompt['attention_mask'],
        'ground_truth_answer': answer 
    }

def get_aime_dataset(tokenizer, version):
    if version == 'AIME_2024':
        AIME_ds = load_dataset("HuggingFaceH4/aime_2024")
    elif version == 'AIME_2025':
        AIME_ds = load_dataset("yentinglin/aime_2025", "default")

    formatted_AIME_ds = AIME_ds.map(
        format_aime, 
        fn_kwargs={'tokenizer': tokenizer},
        remove_columns=AIME_ds['train'].column_names
    )

    return formatted_AIME_ds['train']

def extract_answer(output):
    pattern = r'(?s).*\\boxed\{((?:[^{}]|(?:\{(?:[^{}]|(?R))*\}))*)\}'
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    return None

@torch.no_grad()
def evaluate_result(model, tokenizer, dataloader, k, device, generation_kwargs, stop_token_str: Optional[str] = '<|im_end|>', skip_special_tokens: bool = True):
    num_pass_k = 0 
    num_pass_cons = 0

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    stop_id = tokenizer.convert_tokens_to_ids(stop_token_str)

    for question in dataloader:
        input_ids = torch.tensor(question['input_ids'], device=device).unsqueeze(0)
        attention_mask = torch.tensor(question['attention_mask'], device=device).unsqueeze(0)
        ground_truth = question['ground_truth_answer']

        model_answers = []
        passed = False

        for _ in range(k): 
            model_output_token = model.generate(input_ids, attention_mask=attention_mask, use_cache=True, pad_token_id=pad_id, eos_token_id=stop_id, **generation_kwargs)
            input_length = input_ids.shape[1]
            model_output = tokenizer.decode(model_output_token[0, input_length:], skip_special_tokens=skip_special_tokens)
            model_answer = extract_answer(model_output)
            if model_answer is None: 
                continue
            model_answers.append(model_answer)
            if model_answer == ground_truth and not passed:
                num_pass_k += 1
                passed = True

        if model_answers:
            most_freq_answer = max(set(model_answers), key=model_answers.count)
            if most_freq_answer == ground_truth:
                num_pass_cons += 1

    return {
        'pass@k': num_pass_k / len(dataloader),
        'cons_k': num_pass_cons / len(dataloader)
    }