from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
import torch

model_name = "Qwen3-1.7B-DAPO"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ds = load_dataset("justus27/light-r1-stage-2-sft", split='train')
system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{{}}. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
def format_R1(example):
    prompt = []
    prompt.append({"role": "system", "content": system_prompt})
    prompt.extend(example['messages'])
    return {"messages": prompt}

ds = ds.map(format_R1)

training_args = SFTConfig(
    output_dir="Qwen3-1.7B-DAPO-SFT-stage-2",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    max_seq_length=32768,
    learning_rate=1e-5,
    num_train_epochs=8,
    gradient_checkpointing=True,
    use_liger_kernel=True,
    eos_token='<|im_end|>',
    bf16=True,
    tf32=True,
    torch_compile=True,
    logging_steps=1,
    logging_dir="./logs",  
    logging_strategy="steps",  
    logging_first_step=True,  # Log the first training step
    log_level="info",  # Logging level (info, warning, error, etc)
    save_strategy="epoch", 
    push_to_hub=True,
    hub_model_id="Pi-1905/Qwen3-1.7B-DAPO-SFT-stage-2",
    hub_strategy="all_checkpoints",
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    args=training_args,
    processing_class=tokenizer
)

trainer.train()

# trainer.save_model("Qwen3-0.6B-SFT")
# tokenizer.save_pretrained("Qwen3-0.6B-SFT")