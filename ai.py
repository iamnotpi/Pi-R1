from dataclasses import field
import torch
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
from transformers import DataCollatorWithPadding
from typing import List 
import os
import wandb
import numpy as np 
from tqdm import tqdm 

from data import load_dapo_dataset
from model import Args, load_model_and_tokenizer 

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
half_num_cpu = os.cpu_count() // 2
# torch._functorch.config.activation_memory_budget = 0.25

class GRPOArgs(Args): 
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B" 
    num_rollout_steps: int = 64 # Number of prompts to collect rollouts from per iteration
    grpo_epochs: int = 4         # Number of training epochs on collected data
    mini_batch_size: int = 8    # Number of sequences per training mini-batch
    max_prompt_length: int = 512
    max_generation_length: int = 4096
    gradient_accumulation_steps: int = 4 
    lr: float = 1e-6            # Learning rate for RL
    clip_eps: float = 0.2       # PPO/GRPO clipping epsilon
    num_groups: int = 4         # Number of responses per prompt (G)
    max_iter: int = 1000        # Total training iterations (rollout + train cycles)
    ckpt_step: int = 50
    eval_step: int = 10
    eval_generation_kwargs: dict = field(default_factory=lambda: {
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.95,
        'max_new_tokens': 4096, 
    })

args = GRPOArgs()

def compute_reward(answers: List[str], ground_truths_or_prompts: List[torch.tensor]) -> List[float]:
    pass 


def calculate_grpo_advantages_outcome(rewards, num_groups):
    """ Calculates GRPO advantages using Outcome Supervision"""
    num_prompts = len(rewards) // num_groups
    advantages = []

    for i in range(num_prompts):
        group_rewards = rewards[i * num_groups : (i + 1) * num_groups]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        normalized_rewards = [(r - mean_reward) / (std_reward + 1e-8) for r in group_rewards]
        advantages.extend(normalized_rewards)
    return torch.tensor(advantages)

wandb.init(
    project='qwen-rl', 
    name=f'sft-run-{args.model_name}',
    config=args,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, tokenizer = load_model_and_tokenizer(args.model_name) 
# model = torch.compile(model)
model.gradient_checkpointing_enable()

pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
tokenizer.pad_token_id = pad_token_id
tokenizer.padding_side = 'left' 

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=(device=='cuda'))

dapo_dataset = load_dapo_dataset(tokenizer) 
data_collator = DataCollatorWithPadding(tokenizer, padding='longest', return_tensors='pt')

dataloader = DataLoader(
    dapo_dataset,
    batch_size=1, 
    shuffle=True,
    num_workers=half_num_cpu,
    collate_fn=data_collator
)

data_iter = iter(dataloader)
global_step = 0 

for iter_num in range(args.max_iter):
    print(f"\n--- Iteration {iter_num + 1}/{args.max_iter} ---")

    # Rollouts 
    model.eval() 
    rollout_buffer = {
        'prompt_ids': [], 'prompt_mask': [],        # Store original prompt info
        'full_input_ids': [], 'full_attention_mask': [], # Store prompt + generated sequence
        'log_probs_old': [],                        # Log probs from behavior policy 
        'rewards_raw': [],                          # Raw rewards 
        'prompt_lens': []
    }
    prompts_processed_in_rollout = 0
    pbar = tqdm(total=args.num_rollout_steps, desc=f"Rollout Iter {iter_num+1}", leave=False)
    while prompts_processed_in_rollout < args.num_rollout_steps:
        try:
            batch_question = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_question = next(data_iter)

        input_ids = batch_question['input_ids'].to(device) # (1, prompt_len)
        attention_mask = batch_question['attention_mask'].to(device) # (1, prompt_len)

        if input_ids.shape[1] > args.max_prompt_length:
            continue

        current_prompt_batch_size = input_ids.size(0) # 1
        prompts_processed_in_rollout += current_prompt_batch_size
        global_step += current_prompt_batch_size

        model.config.use_cache = True

        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=args.num_groups,
                output_scores=True, 
                return_dict_in_generate=True,
                max_new_tokens=args.max_generation_length,
                pad_token_id=pad_token_id, 
                eos_token_id=tokenizer.eos_token_id, 
                **args.eval_generation_kwargs
            )

        model.config.use_cache = True

        sequences = outputs.sequences # (G, full_len), includes prompt + generated + padding
        scores = torch.stack(outputs.scores, dim=1) # (G, gen_len, vocab_size)

        actual_prompt_len = input_ids.shape[1] 
        gen_part = sequences[:, actual_prompt_len:] 
        gen_len_actual = gen_part.shape[1] 

        dist = Categorical(logits=scores)
        action_log_probs = dist.log_prob(gen_part) # (G, gen_len)

        # Mask out padding log_probs
        gen_mask = (gen_part != pad_token_id).long() # (G, gen_len)
        action_log_probs = action_log_probs * gen_mask 

        # Compute rewards 
        decoded_outputs = tokenizer.batch_decode(gen_part, skip_special_tokens=True) 
        current_rewards = compute_reward(decoded_outputs) # List of G rewards

        rollout_buffer['prompt_ids'].append(input_ids.cpu())
        rollout_buffer['prompt_mask'].append(attention_mask.cpu())
        rollout_buffer['full_input_ids'].append(sequences.cpu()) 
        rollout_buffer['full_attention_mask'].append((sequences != pad_token_id).long().cpu())
        rollout_buffer['log_probs_old'].append(action_log_probs.cpu())
        rollout_buffer['rewards_raw'].extend(current_rewards) # Append rewards for this group
        rollout_buffer['prompt_lens'].extend([actual_prompt_len] * args.num_groups)

        pbar.update(current_prompt_batch_size)

    pbar.close()
    # --- Collate and Process Rollout Data ---

    max_len_in_batch = max(seq.shape[1] for seq in rollout_buffer['full_input_ids'])

    padded_full_ids = []
    padded_full_mask = []
    padded_log_probs = []
    max_gen_len_in_batch = max_len_in_batch - min(rollout_buffer['prompt_lens'])

    num_sequences = len(rollout_buffer['full_input_ids'])

    seq_dtype = rollout_buffer['full_input_ids'][0].dtype
    logp_dtype = rollout_buffer['log_probs_old'][0].dtype

    all_full_ids = torch.full((num_sequences, max_len_in_batch), pad_token_id, dtype=seq_dtype, device='cpu')
    all_full_mask = torch.zeros((num_sequences, max_len_in_batch), dtype=torch.long, device='cpu')
    all_log_probs_old_padded = torch.zeros((num_sequences, max_gen_len_in_batch), dtype=logp_dtype, device='cpu')

    for i in range(num_sequences):
        seq = rollout_buffer['full_input_ids'][i]
        mask = rollout_buffer['full_attention_mask'][i]
        log_p = rollout_buffer['log_probs_old'][i]

        seq_len = seq.shape[1]
        log_p_len = log_p.shape[1]

        all_full_ids[i, :seq_len] = seq.view(-1) 
        all_full_mask[i, :seq_len] = mask.view(-1) 

        if log_p_len > 0:
            copy_len = min(log_p_len, max_gen_len_in_batch)
            all_log_probs_old_padded[i, :copy_len] = log_p.view(-1)[:copy_len] 

    all_prompt_lens = torch.tensor(rollout_buffer['prompt_lens'], dtype=torch.long, device='cpu')
    all_rewards_raw = rollout_buffer['rewards_raw'] 

    # Calculate advantages
    all_advantages_raw = calculate_grpo_advantages_outcome(all_rewards_raw, args.num_groups)

    avg_raw_rewards = np.mean(all_rewards_raw)
    wandb.log({'rollout/avg_raw_reward': avg_raw_rewards}, step=global_step)

    num_sequences_rollout = all_full_ids.size(0)
    max_gen_len_rollout = all_log_probs_old_padded.size(1) # Use the actual max gen len from padded log_probs
    advantages_padded = all_advantages_raw.unsqueeze(1).repeat(1, max_gen_len_rollout).cpu()

    indices = np.arange(num_sequences_rollout) 

    for epoch in range(args.grpo_epochs):
        print(f"  Epoch {epoch + 1}/{args.grpo_epochs}")
        np.random.shuffle(indices) 
        epoch_total_loss = 0.0
        epoch_policy_loss = 0.0

        # Mini-batch loop
        for i in range(0, num_sequences_rollout, args.mini_batch_size):
            model.train()

            batch_indices = indices[i : i + args.mini_batch_size]

            # Get mini-batch data 
            mb_full_ids = all_full_ids[batch_indices].to(device)
            mb_full_mask = all_full_mask[batch_indices].to(device)
            mb_log_probs_old_padded = all_log_probs_old_padded[batch_indices].to(device)
            mb_advantages_padded = advantages_padded[batch_indices].to(device)
            mb_prompt_lens = all_prompt_lens[batch_indices].to(device)

            mb_seq_len = mb_full_ids.shape[1]
            position_ids = torch.arange(mb_seq_len, device=device).unsqueeze(0).expand_as(mb_full_ids)
            # Mask is 1 if position >= prompt_len AND original mask is 1
            mb_gen_mask = (position_ids >= mb_prompt_lens.unsqueeze(1)) & (mb_full_mask > 0)

            # Select only the valid *generated* tokens using the mask
            mb_log_probs_old = mb_log_probs_old_padded[mb_gen_mask[:, -max_gen_len_rollout:]] # Slice padded logprobs by gen mask
            mb_advantages = mb_advantages_padded[mb_gen_mask[:, -max_gen_len_rollout:]]    # Slice padded advantages by gen mask

            # Get the actions (token ids) that correspond to these logprobs/advantages
            # Need to apply the gen_mask to the full_ids to get the right tokens
            mb_actions = mb_full_ids[mb_gen_mask]

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=mb_full_ids, attention_mask=mb_full_mask)
                logits = outputs.logits
                valid_gen_logits = logits.view(-1, logits.size(-1))[mb_gen_mask.view(-1)] # (num_valid_tokens, vocab_size)

                dist = Categorical(logits=valid_gen_logits)
                mb_log_probs_new = dist.log_prob(mb_actions)

                logratio = mb_log_probs_new - mb_log_probs_old
                ratio = torch.exp(logratio)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2)) 

                loss = pg_loss 

            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            epoch_total_loss += loss.item()
            epoch_policy_loss += pg_loss.item()

            num_batches_processed = (i // args.mini_batch_size) + 1
            if num_batches_processed % args.gradient_accumulation_steps == 0 or (i + args.mini_batch_size) >= num_sequences_rollout:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step()
                optimizer.zero_grad()

                # Note: Loss logging here is approximate average per update step
                avg_loss_accum = epoch_total_loss / (num_batches_processed * args.mini_batch_size)
                avg_pg_loss_accum = epoch_policy_loss / (num_batches_processed * args.mini_batch_size)
                wandb.log({
                    "train/step_loss": avg_loss_accum,
                    "train/step_policy_loss": avg_pg_loss_accum,
                    "train/grad_norm": grad_norm.item() if grad_norm is not None else 0.0
                }, step=global_step) 

        num_update_steps_epoch = (num_sequences_rollout + args.mini_batch_size - 1) // args.mini_batch_size
        avg_epoch_loss = epoch_total_loss / num_update_steps_epoch
        avg_epoch_pg_loss = epoch_policy_loss / num_update_steps_epoch
        print(f"  Avg Epoch Loss: {avg_epoch_loss:.4f} | Avg Policy Loss: {avg_epoch_pg_loss:.4f}")
        wandb.log({
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch_policy_loss": avg_epoch_pg_loss,
            "epoch": iter_num * args.grpo_epochs + epoch 
        }, step=global_step)

    if (iter_num + 1) % args.ckpt_step == 0:
        save_dir = f"checkpoints/grpo_{args.model_name.split('/')[-1]}_iter_{iter_num+1}"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Checkpoint saved to {save_dir}")

    # Evaluation
    # if (iter_num + 1) % args.eval_step == 0:
    #     model.eval()

wandb.finish()