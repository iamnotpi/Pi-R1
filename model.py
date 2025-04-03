from dataclasses import dataclass, field
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math 


@dataclass
class Args:
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B"
    max_iter: int = 10000
    save_new_models: bool = False
    batch_size: int = 1
    max_prompt_length: int = 4096
    max_generation_length: int = 8192
    context_length: int = 32768
    num_epochs: int = 4 
    num_groups: int = 4 # GRPO thing
    max_lr: float = 2.5e-4
    gradient_accumulation_steps: int = 16 
    use_compile: bool = False
    ckpt_step: int = 1000 
    eval_step: int = 100 
    k: int = 5 # pass@k 
    eval_generation_kwargs: dict = field(default_factory=lambda: {
        'do_sample': True,
        'temperature': 1.0,
        'top_p': 0.95,
        'max_new_tokens': 4096,
    })


def load_model_and_tokenizer(model_name, use_pretrained=True, model_path=None, load_model=True, load_tokenizer=True, save_new_models=False):
    """
    Load model, tokenizer and save new model + tokenizer with new tokens(optional)
    """
    model = None
    tokenizer = None

    if use_pretrained:
        if load_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            new_tokens = ['<think>', '</think>', '<answer>', '</answer>']
            tokenizer.add_tokens(new_tokens)
            
        if load_model: 
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
            model.resize_token_embeddings(len(tokenizer))

    else:
        if load_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        if load_model: 
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )

    if save_new_models:
        tokenizer.save_pretrained(f"{model_name}-Extended")
        model.save_pretrained(f"{model_name}-Extended")

    return model, tokenizer

def lr_scheduler(step, warm_up_step, max_decay_step, max_lr, min_lr):
    if step < warm_up_step: 
        lr = max_lr * (step + 1) / warm_up_step
    elif step < max_decay_step:
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi / 2 * (step - warm_up_step) / (max_decay_step - warm_up_step)))
    else:
        lr = min_lr
    return lr

# def generate_one_completion(input_data, max_new_tokens, im_end_id, eos_token_id, input_type='text', stop_at_im_end=True, skip_special_tokens=True):
#     eos_params = [im_end_id] if stop_at_im_end else None
#     if input_type == 'text':
#         inputs = tokenizer(input_data, return_tensors="pt")
#         inputs = {k: v.to(model.device) for k, v in inputs.items()}
#         output = model.generate(**inputs, max_new_tokens=max_new_tokens, eos_token_id=eos_params, pad_token_id=eos_token_id)
#     elif input_type == 'tensors':
#         inputs = input_data['input_ids']
#         attention_mask = input_data['attention_mask']
#         output = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=max_new_tokens, eos_token_id=eos_params, pad_token_id=eos_token_id)
#     return tokenizer.decode(output[0][inputs['input_ids'].numel():], skip_special_tokens=skip_special_tokens)