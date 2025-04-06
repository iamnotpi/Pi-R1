import os 
import math 
import time 
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding 
import wandb

from data import load_light_r1_sft_dataset, load_aime_dataset
from eval import evaluate_result
from model import Args, load_model_and_tokenizer, lr_scheduler

# Mitigate memory fragmentation 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def custom_collate_fn_wrapper(batch, default_collator: DataCollatorWithPadding):
    numerical_features = []
    ground_truth_answers = []
    for item in batch:
        numerical_item = {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask']
        }
        numerical_features.append(numerical_item)
        ground_truth_answers.append(item['ground_truth_answer'])
    padded_batch = default_collator(numerical_features)
    padded_batch['ground_truth_answer'] = ground_truth_answers
    return padded_batch

args = Args(
    model_name="Qwen/Qwen2.5-Coder-1.5B",
    context_length=16384, 
    batch_size=1, 
    use_compile=False,
    gradient_accumulation_steps=64,
    ckpt_step=250,
    eval_step=50,
    k=5,
    eval_generation_kwargs={
        'do_sample': True,
        'temperature': 1.0, 
        'top_p': 0.95,
        'max_new_tokens': 4096,
    }
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

if args.use_compile:
    model = torch.compile(model)

model.train()

SFT_dataset = load_light_r1_sft_dataset(tokenizer, context_length=args.context_length, num_proc=half_num_cpu)
aime_dataset = load_aime_dataset(tokenizer, 'AIME_2024')

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
inf_data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

dataloader = DataLoader(
    SFT_dataset, 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=half_num_cpu,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=data_collator
)
aime_dataloader = DataLoader(
    aime_dataset, 
    batch_size=1, 
    shuffle=False,
    num_workers=half_num_cpu,
    pin_memory=True,
    collate_fn=lambda batch: custom_collate_fn_wrapper(batch, default_collator=inf_data_collator)  
)

optimizer = torch.optim.AdamW(model.parameters(), fused=True)

dataloader_len = len(dataloader)

# lr scheduler config
total_steps = math.ceil(args.num_epochs * dataloader_len / args.gradient_accumulation_steps)
max_decay_step = total_steps 
warm_up_step = int(0.1 * max_decay_step)

torch.set_float32_matmul_precision('high')

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

        if global_step > 0 and global_step % args.eval_step == 0 or global_step == total_steps:
            aime_res = evaluate_result(model=model, 
                                       tokenizer=tokenizer, 
                                       dataloader=aime_dataloader, 
                                       k=args.k, 
                                       device=device, 
                                       generation_kwargs=args.eval_generation_kwargs
            )
            writer.add_scalar('Evaluation/AIME24_pass@k', aime_res['pass@k'], global_step)
            writer.add_scalar('Evaluation/AIME24_cons@k', aime_res['cons_k'], global_step)
    
writer.close()
wandb.finish()