import regex as re 
import torch 
from typing import Optional

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