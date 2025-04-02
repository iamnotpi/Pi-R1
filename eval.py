from datasets import load_dataset
import regex as re 
import torch 
from typing import Optional


def format_aime(question, tokenizer): 
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
    model.eval()

    num_pass_k = 0 
    num_pass_cons = 0
    num_questions = 0

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    stop_id = tokenizer.convert_tokens_to_ids(stop_token_str)

    eval_gen_kwargs = generation_kwargs.copy()

    eval_gen_kwargs['num_return_sequences'] = k
    eval_gen_kwargs['pad_token_id'] = pad_id
    eval_gen_kwargs['eos_token_id'] = [stop_id]

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ground_truths = batch['ground_truth_answer']

        current_batch_size = input_ids.shape[0]

        model_output_tokens = model.generate(input_ids, attention_mask=attention_mask, use_cache=True, **eval_gen_kwargs)
        input_lengths = torch.sum(attention_mask, dim=1)

        generated_token_ids = []

        for i in range(current_batch_size * k): 
            original_prompt_idx = i // k
            original_input_length = input_lengths[original_prompt_idx]
            generated_ids = model_output_tokens[i, original_input_length:]
            generated_token_ids.append(generated_ids)

        decoded_outputs = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
        
        batch_pass_k = 0
        batch_cons_k = 0

        for i in range(current_batch_size):
            start_idx = i * k
            end_idx = start_idx + k
            question_outputs = decoded_outputs[start_idx:end_idx]
            current_ground_truth = ground_truths[i]

            model_answers_for_question = []
            passed = False

            for output_str in question_outputs: 
                model_answer = extract_answer(output_str)
                if model_answer is None:
                    continue

                model_answers_for_question.append(model_answer)

                if not passed and model_answer == current_ground_truth:
                    batch_pass_k += 1
                    passed = True

            if model_answers_for_question:
                most_freq_answer = max(set(model_answers_for_question), key=model_answers_for_question.count)
                if most_freq_answer == current_ground_truth:
                    batch_cons_k += 1

        num_pass_k += batch_pass_k
        num_pass_cons += batch_cons_k
        num_questions += current_batch_size
        
    return {
        'pass@k': num_pass_k / num_questions,
        'cons_k': num_pass_cons / num_questions
    }