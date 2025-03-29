from datasets import load_dataset
import os 
import math 
import torch 
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
from model import Args, load_model_and_tokenizer
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorForLanguageModeling
import wandb

args = Args(
    model_name="Qwen/Qwen2.5-Coder-0.5B",
    context_length=16384, 
    batch_size=1 
)
half_num_cpu = os.cpu_count() // 2

wandb.init(
    project='qwen-sft', 
    name=f'sft-run-{args.model_name}',
    config=args,
    sync_tensorboard=True
)

writer = SummaryWriter()

model, tokenizer = load_model_and_tokenizer(args.model_name)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
using_cuda = device == 'cuda'
optimizer = torch.optim.AdamW(model.parameters(), fused=using_cuda)
dtype = torch.bfloat16 if using_cuda else torch.float32

# lr scheduler config
max_decay_step = len(dataloader) * args.num_epochs 
warm_up_step = int(0.2 * max_decay_step)

torch.set_float32_matmul_precision('high')

if using_cuda:
    model = torch.compile(model)

model.train()

for epoch in range(args.num_epochs):
    for batch_idx, batch in enumerate(dataloader):  
        step = epoch * len(dataloader) + batch_idx
        
        lr = lr_scheduler(
            step=step,
            warm_up_step=warm_up_step,
            max_decay_step=max_decay_step,
            max_lr=args.max_lr,
            min_lr=0.1*args.max_lr
        )

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        X = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        B, T = X.shape

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            y_preds = model(X, attention_mask=mask).logits
            loss = F.cross_entropy(y_preds.view(B * T, -1), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward() 
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), step)
        writer.add_scalar('Chart/norm', norm.item(), step)
        writer.add_scalar('Chart/lr', lr, step)
    
writer.close()
wandb.finish()