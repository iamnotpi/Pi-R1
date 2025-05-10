from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import re

dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split='train')

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

training_args = GRPOConfig(
    output_dir="Qwen3-1.7B-GRPO", 
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.8,
    vllm_max_model_len=9216,
    max_completion_length=8192,
    learning_rate=1e-6,
    epsilon=0.2,
    epsilon_high=0.28,
    # scale_rewards=False,
    # loss_type="dr_grpo",
    beta=0.0,
    # mask_truncated_completions=True,
    num_generations=8,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    torch_compile=True,
    num_train_epochs=1,
    logging_first_step=True,
    logging_steps=10,
    save_steps=100,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B-Base",
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()