from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

from huggingface_hub import login
login(token="your_personal_access_token")

dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split='train')

system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{{}}. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. The assistant MUST use English only."

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]

def normalize_final_answer(final_answer):
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Simplify LaTeX
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers 
    try:
        if final_answer.replace(",", "").isdigit():
            final_answer = final_answer.replace(",", "")
    except AttributeError:
        pass

    return final_answer.strip()

def reward_func(completions, solution, **kwargs):
    rewards = []
    answer_pattern = r"(?i)Answer\s*:\s*([^\n]+)"

    for completion, gold in zip(completions, solution):
        matches = re.findall(answer_pattern, completion)
        extracted_answer = matches[-1].strip() if matches else "[INVALID]" 

        norm_pred = normalize_final_answer(extracted_answer)
        norm_gt = normalize_final_answer(str(gold))
        is_correct = (norm_pred == norm_gt) and norm_pred != "[INVALID]"
        rewards.append(1.0 if is_correct else 0.0)

    return rewards

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[ExprExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                reward = 0.0
        else:
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards

def format_R1(example):
    prompt = []
    prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": example["prompt"]})
    return {"prompt": prompt}

dataset = dataset.map(format_R1)

training_args = GRPOConfig(
    output_dir="Qwen3-1.7B-GRPO", 
    per_device_train_batch_size=8,
    gradient_accumulation_steps=6,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.85,
    vllm_max_model_len=4096+1568,
    max_prompt_length=512,
    max_completion_length=4096+1024,
    learning_rate=1e-6,
    epsilon=0.2,
    epsilon_high=0.28,
    # scale_rewards=False,
    # loss_type="dr_grpo",
    beta=0.0,
    # mask_truncated_completions=True,
    num_generations=6,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    torch_compile=True,
    num_train_epochs=1,
    log_level="info",
    logging_first_step=True,
    logging_steps=1,
    save_steps=5,
    report_to="wandb",
    push_to_hub=True,
    hub_model_id="Pi-1905/Qwen3-1.7B-DAPO",
    hub_strategy="all_checkpoints",
    # lr_scheduler_type='cosine_with_min_lr',
    # lr_scheduler_kwargs={'min_lr': 1e-6},
    warmup_ratio=0.1
)

model_name = "Qwen/Qwen3-1.7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=accuracy_reward,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset,
)

trainer.train(resume_from_checkpoint="Pi-1905/Qwen3-1.7B-DAPO")