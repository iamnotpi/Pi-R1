import torch
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorWithPadding 
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
    eval_generation_kwargs={
        'do_sample': True,
        'temperature': 1.0, 
        'top_p': 0.95,
        'max_new_tokens': 4096
    }
)

def extract_answer(answer):
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

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

dapo_dataset = load_dapo_dataset()
data_collator = DataCollatorWithPadding(tokenizer, padding='longest')

dataloader = DataLoader(
    dapo_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=half_num_cpu,
    collate_fn=data_collator
)

# Pre-allocate tensors 
states = torch.zeros((args.batch_size, args.num_groups, args.max_prompt_length + args.max_generation_length))
actions = torch.zeros((args.batch_size, args.num_groups, args.max_generation_length))
advantages = torch.zeros((args.batch_size, args.num_groups, args.max_generation_length))
log_probs = torch.zeros((args.batch_size, args.num_groups, args.max_generation_length))
rewards = torch.zeros((args.batch_size, args.num_groups))
prompt_lengths = torch.zeros(args.batch_size)
response_lengths = torch.zeros((args.batch_size, args.num_groups))

# Rollouts
for batch_question in dataloader:
    input_ids = batch_question['input_ids']
    attention_mask = batch_question['attention_mask']
    ground_truth = batch_question['ground_truth']

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask, 
        output_scores=True,
        return_dict_in_generate=True,
        max_length=args.max_prompt_length + args.max_generation_length,
        num_return_sequences=args.num_groups,
        do_sample=True,
        temperature=0.7
    )

    sequences = outputs.sequences # (B, seq_len) (seq_len = max_length (2048?) + padded_prompt_length)
    scores = torch.stack(outputs.scores, dim=-1).transpose(-1, -2) # (*(B, vocab_size)) -> (B, vocab_size, max_len) -> (B, max_len, vocab_size)

    prompt_lengths = attention_mask.sum(dim=1)

    for i in range(args.batch_size): 
        prompt_length = prompt_lengths[i].item() 
        actions_length = min(args.max_generation_length, len(sequences[i]) - prompt_length)
        current_action = sequences[i, prompt_length:prompt_length + actions_length] # (1, action_length)
        states[i, :prompt_length + actions_length] = sequences[i, :prompt_length + actions_length]
        actions[i, :actions_length] = current_action

        dist = Categorical(logits=scores[i, :actions_length])
        log_probs[i, :actions_length] = dist.log_prob(current_action)

    rewards = torch.zeros((args.batch_size,)).to(device)


    # TODO:
    # Metrics: 
    # 1. Reward
    # 2. Response length
    # 3. Entropy 
    # 4.  

writer.close()
wandb.finish()