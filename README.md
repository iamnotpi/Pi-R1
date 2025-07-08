# Finetuning Qwen 3 1.7B with DAPO

This project focuses on fine-tuning [Qwen 3 1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) using the [DAPO](https://arxiv.org/abs/2503.14476) method on the [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) dataset.

📍 **Checkpoints**:  
All resulting checkpoints are publicly available here:  
👉 [https://huggingface.co/Pi-1905/Qwen3-1.7B-DAPO/tree/main](https://huggingface.co/Pi-1905/Qwen3-1.7B-DAPO/tree/main)

---

## 🛠️ Environment Setup

Install all required dependencies via:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training Instructions

Training is performed on a machine with **8×H100 GPUs** using `transformers`, `trl`, and `accelerate`.

### Step 1: Launch the vLLM server

Use the first 2 GPUs for inference service:

```bash
CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve --model Qwen/Qwen3-1.7B --data_parallel_size 2
```

### Step 2: Run the DAPO trainer

Use the remaining 6 GPUs for training:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file config/zero2.yaml --num_processes 6 \
rl_trainer.py
```

---

## 📈 Evaluation

We evaluate the fine-tuned model using [LightEval](https://github.com/huggingface/lighteval).

Run the following command to evaluate the checkpoint:

```bash
lighteval vllm "model_name=Qwen3-1.7B-DAPO/checkpoint-30,dtype=bfloat16,tensor_parallel_size=2,max_model_length=34000,max_num_batched_tokens=32768,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95,top_k:20,min_p:0}" \
"lighteval|aime24|0|0" \
--use-chat-template
```
---
