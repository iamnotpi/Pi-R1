import torch
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorWithPadding 
from typing import List
import os
import wandb

from data import load_dapo_dataset
from model import Args, load_model_and_tokenizer

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
half_num_cpu = os.cpu_count() // 2

args = Args(
    model_name="Qwen/Qwen2.5-Coder-1.5B",
    batch_size=1, 
    use_compile=False,
    gradient_accumulation_steps=64,
    ckpt_step=250,
    eval_step=50,
    k=5,
    max_generation_length=8192,
    max_prompt_length=512,
    num_updates=32,
    eval_generation_kwargs={
        'do_sample': True,
        'temperature': 1.0, 
        'top_p': 0.95,
        'max_new_tokens': 4096
    },
)

args.mini_batch_size = args.num_groups * args.max_generation_length // args.num_updates

def compute_reward(answers: List[str], ground_truths: torch.tensor): 
    # TODO: compute reward using math-verify
    pass 

wandb.init(
    project='qwen-rl', 
    name=f'sft-run-{args.model_name}',
    config=args,
    sync_tensorboard=True
)

writer = SummaryWriter()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, tokenizer = load_model_and_tokenizer(args.model_name)

pad_token_id = tokenizer.pad_token_id = tokenizer.eos_token_id

optimizer = torch.optim.AdamW(model.parameters(), fused=(device=='cuda'))

dapo_dataset = load_dapo_dataset(tokenizer)
data_collator = DataCollatorWithPadding(tokenizer, padding='longest')

dataloader = DataLoader(
    dapo_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=half_num_cpu,
    collate_fn=data_collator
)

# Pre-allocate tensors 
states_and_actions = torch.full((args.batch_size * args.num_groups, args.max_prompt_length + args.max_generation_length), pad_token_id)
advantages = torch.zeros((args.batch_size * args.num_groups, args.max_generation_length))
log_probs = torch.zeros((args.batch_size * args.num_groups, args.max_generation_length))
rewards = torch.zeros((args.batch_size * args.num_groups))
prompt_lengths = torch.zeros(args.batch_size)
response_lengths = torch.zeros((args.batch_size * args.num_groups))

for iter in range(args.max_iter):
    # Rollouts
    for batch_question in dataloader:
        input_ids = batch_question['input_ids'].to(device)
        attention_mask = batch_question['attention_mask'].to(device)
        ground_truths = batch_question['ground_truth'].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                output_scores=True, # scores = logits / temperature 
                return_dict_in_generate=True,
                max_new_tokens=args.max_generation_length,
                num_return_sequences=args.num_groups,
                do_sample=True,
                temperature=0.7,
                eos_token_id=[pad_token_id]
            )

        sequences = outputs.sequences # (args.batch_size * args.num_groups, padded_prompt_len + max_gen_len) 
        scores = torch.stack(outputs.scores, dim=1) # (*(B, vocab_size)) -> (B, max_len, vocab_size)

        prompt_lengths = attention_mask.sum(dim=1)

        prompt_len_padded = input_ids.shape[1] # Length of the longest prompt
        gen_part = sequences[:, prompt_len_padded:] # (args.batch_size * args.num_groups, max_gen_len) 
        gen_len_padded = gen_part.shape[1] # Length of the longest answer

        states_and_actions[:, args.max_prompt_length - prompt_len_padded:args.max_prompt_length] = input_ids
        states_and_actions[:, args.max_prompt_length:args.max_prompt_length + gen_len_padded] = gen_part.view(args.batch_size * args.num_groups, -1)

        prompt_lengths = attention_mask.sum(dim=1)

        generation_mask = (gen_part != pad_token_id).type(torch.int)
        response_lengths = generation_mask.sum(dim=-1)

        total_mask = (sequences != pad_token_id).type(torch.int)
        
        dist = Categorical(logits=scores)
        action_log_probs = dist.log_prob(gen_part)
        log_probs[:, :gen_len_padded] = action_log_probs

        # Rewards
        output_sequences = tokenizer.batch_decode(gen_part, skip_special_tokens=True)
        rewards = torch.tensor(compute_reward(output_sequences, ground_truths), device=device)
        rewards = rewards.view(args.batch_size, args.num_groups)
        advantages[:, :gen_len_padded] = (rewards - rewards.mean(dim=1, keepdim=True)) / rewards.std(dim=1, keepdim=True) # batch_size * num_groups

    # Training
    for _ in range(args.batch_size): # 1 question
        start = 0
        ids = torch.randperm(args.batch_size * args.num_groups)
        for mini_batch_size in range(args.mini_batch_size):
            end = start + mini_batch_size
            idx = ids[start:end]

            mb_states_and_actions = states_and_actions[idx]
            mb_advantages = advantages[idx]

            with torch.autocast('cuda', dtype=torch.bfloat16):
                new_logits = model(input_ids=mb_states_and_actions, attention_mask=total_mask).logits
            new_probs = Categorical(logits=new_logits[args.max_prompt_length:])
            new_log_probs = new_probs.log_prob(mb_states_and_actions[args.max_prompt_length:])

            ratio = torch.exp(new_log_probs - log_probs[idx])

            loss = -torch.mean(torch.sum(min(ratio * mb_advantages, torch.clip(ratio, 1 - args.low_eps, 1 + args.high_eps) * mb_advantages), dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            start = end

        # TODO:
        # Metrics: 
        # 1. Reward
        # 2. Response length 
        # 3. Entropy 

writer.close()
wandb.finish()