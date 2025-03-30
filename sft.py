from datasets import load_dataset
import os 
import math 
import time 
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import Args, load_model_and_tokenizer
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorForLanguageModeling
import wandb

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

args = Args(
    model_name="Qwen/Qwen2.5-Coder-0.5B",
    context_length=16384, 
    batch_size=1, 
    use_compile=False,
    gradient_accumulation_steps=64
)
half_num_cpu = os.cpu_count() // 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device == 'cuda', 'GPU is required.'

wandb.init(
    project='qwen-sft', 
    name=f'sft-run-{args.model_name}',
    config=args,
    sync_tensorboard=True
)

writer = SummaryWriter()

model, tokenizer = load_model_and_tokenizer(args.model_name)
model.gradient_checkpointing_enable()

ds = load_dataset("qihoo360/Light-R1-SFTData")

def format_and_tokenize(conversations, max_length): 
    dialogue = """<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<|im_end|>\n"""
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

SFT_dataset = ds.map(
    lambda x: format_and_tokenize(x["conversations"], max_length=args.context_length),
    remove_columns=['conversations'],
    num_proc=half_num_cpu
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

dataloader = DataLoader(
    SFT_dataset['train'], 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=half_num_cpu,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=data_collator
)

def lr_scheduler(step, warm_up_step, max_decay_step, max_lr, min_lr):
    if step < warm_up_step: 
        lr = max_lr * (step + 1) / warm_up_step
    elif step < max_decay_step:
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi / 2 * (step - warm_up_step) / (max_decay_step - warm_up_step)))
    else:
        lr = min_lr
    return lr

optimizer = torch.optim.AdamW(model.parameters(), fused=True)

dataloader_len = len(dataloader)

# 4 / 64 * 79000
# lr scheduler config
total_steps = math.ceil(args.num_epochs * dataloader_len / args.gradient_accumulation_steps)
max_decay_step = total_steps 
warm_up_step = int(0.1 * max_decay_step)

torch.set_float32_matmul_precision('high')

if args.use_compile:
    model = torch.compile(model)

model.train()

global_step = 0
idx = 0
accum_loss = 0 

os.makedirs('checkpoints', exist_ok=True)

for epoch in range(args.num_epochs):
    # Gradients accumulation 
    optimizer.zero_grad() 

    start = time.time()
    batch_tokens = 0

    for _, mini_batch in enumerate(dataloader):   
        X = mini_batch["input_ids"].to(device)
        mask = mini_batch["attention_mask"].to(device)
        y = mini_batch["labels"].to(device)

        B, T = X.shape
        batch_tokens += B * T 

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            y_preds = model(X, attention_mask=mask).logits
            loss = F.cross_entropy(y_preds.view(B * T, -1), y.view(-1))

        accum_loss += loss.item()
        loss /= args.gradient_accumulation_steps
        loss.backward()  

        idx += 1
        
        is_update = (idx + 1) % args.gradient_accumulation_steps == 0
        is_last = idx == dataloader_len - 1

        if is_update or is_last: 
            global_step += 1
            lr = lr_scheduler(
                step=global_step,
                warm_up_step=warm_up_step,
                max_decay_step=max_decay_step,
                max_lr=args.max_lr,
                min_lr=0.1*args.max_lr
            )

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            end = time.time()
            if is_update:
                accum_loss /= args.gradient_accumulation_steps
            else:
                accum_loss /= idx % args.gradient_accumulation_steps + 1

            tok_per_sec = batch_tokens / (end - start)

            writer.add_scalar('Loss/train', accum_loss, global_step)
            writer.add_scalar('Charts/norm', norm.item(), global_step)
            writer.add_scalar('Charts/lr', lr, global_step)
            writer.add_scalar('Performance/time_per_batch', end - start, global_step)
            writer.add_scalar('Performance/token_per_sec', tok_per_sec, global_step)

            accum_loss = 0
            batch_tokens = 0
            start = time.time()
        
        if global_step > 0 and global_step % args.ckpt_step == 0 or global_step == total_steps: 
            torch.save(model.state_dict(), f'checkpoints/Qwen-step-{global_step}.pt')
    
writer.close()
wandb.finish()