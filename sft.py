from datasets import load_dataset
import torch 
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import Args, load_model_and_tokenizer
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorForLanguageModeling

writer = SummaryWriter()

args = Args(model_name="Qwen/Qwen2.5-Coder-0.5B")
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
    num_proc=3
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

dataloader = DataLoader(
    SFT_dataset['train'], 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=data_collator
)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if device == 'cuda' else torch.float32

torch.set_float32_matmul_precision('high')

if device == 'cuda':
    model = torch.compile(model)

for epoch in range(args.num_epochs):
    for batch_idx, (X, mask, y) in enumerate(dataloader):
        X = X.to(device)
        mask = mask.to(device)
        y = y.to(device)

        B, T = X.shape

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            y_preds = model(X, attention_mask=mask).logits
            loss = F.cross_entropy(y_preds.view(B * T, -1), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)

writer.close()