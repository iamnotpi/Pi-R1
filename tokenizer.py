import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", padding_size='left')
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B",
    output_hidden_states=True
).to(device)

new_tokens = ['<think>', '</think>', '<answer>', '</answer>']

tokenizer.add_tokens(new_tokens)