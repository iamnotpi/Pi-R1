from datasets import load_dataset
# from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
from trl import GRPOConfig, GRPOTrainer
import re

dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split='train')

def extract_answer(completion):
    if not completion:
        return None

    # Split into lines and remove leading/trailing whitespace from the whole completion
    lines = completion.strip().splitlines()

    last_line = ""
    for line in reversed(lines):
        stripped_line = line.strip()
        if stripped_line:
            last_line = stripped_line
            break

    if not last_line:
        return None # No non-empty lines found

    # Regex explanation:
    # ^         - Start of the string (the last line)
    # Answer:   - Literal "Answer:" (case-insensitive due to flag)
    # \s*       - Zero or more whitespace characters
    # (.*)      - Capture group 1: any character (.), zero or more times (*)
    # $         - End of the string
    match = re.match(r"Answer:\s*(.*)$", last_line, re.IGNORECASE)

    if match:
        # Extract the captured group (the actual answer) and strip any surrounding whitespace
        answer = match.group(1).strip()
        return answer
    else:
        return None

def reward_func(completions, solution, **kwargs): 
    answers = [extract_answer(completion) for completion in completions]
    return [1.0 if answer == str(gold) else 0.0 for answer, gold in zip(answers, solution)]

# def reward_func_math_verify(completions, reward_model, **kwargs):
#     return [float(verify(
#         parse(
#             gold['ground_truth'], 
#             extraction_config=(
#                 LatexExtractionConfig(boxed_match_priority=0),
#                 ExprExtractionConfig(),
#             ),
#         ),
#         parse(
#             completion[0]['content'], 
#             extraction_config=(
#                 LatexExtractionConfig(boxed_match_priority=0),
#                 ExprExtractionConfig(),
#             ),
#         )
#     )) for gold, completion in zip(reward_model, completions)]
    
training_args = GRPOConfig(
    output_dir="Qwen3-0.6B-GRPO", 
    logging_steps=10,
    max_completion_length=4096,
    epsilon=0.2,
    learning_rate=5e-6,
    # epsilon_high=0.28,
    # scale_rewards=False,
    # loss_type="dr_grpo",
    beta=0.0,
    # mask_truncated_completions=True,
    num_generations=4,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    torch_compile=True
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B-Base",
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()