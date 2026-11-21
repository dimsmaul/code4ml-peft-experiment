"""LoRA fine-tuning of another ~7B base model on Code4ML, one seed, with exactly the hyperparameters of the Qwen2.5-Coder-7B
LoRA arm (training/lora/index.ipynb). Justification POC: is the weak result specific to Qwen?

    python train_lora.py deepseek-6.7b
    python train_lora.py seed-coder-8b

Base model and adapter repo come from the arm <key>-lora in run_arm.py (written next to this file by the notebook).
Pushes adapter_* + metrics.json (no tokenizer: vLLM would use it instead of the base one) to <repo>/seed-42/; a model
already on the Hub is skipped. Exit code != 0 on failure.
"""
import json
import sys
import time
import traceback
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

from run_arm import ARMS

DATASET, DATASET_CONFIG = "dimsmauls/code4ml-annotated-rated", "training"
SEED = 42
# ---- identical to training/lora/index.ipynb ----
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 2e-4
EPOCHS = 3
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 4          # effective batch 16
MAX_LENGTH = 2048
OPTIMIZER = "adamw_bnb_8bit"


def main(key):
    arm = ARMS[f"{key}-lora"]
    base, repo = arm["base"], arm["adapter"]
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    if f"seed-{SEED}/adapter_model.safetensors" in api.list_repo_files(repo):
        print(f"{repo}/seed-{SEED} already on the Hub, skipping")
        return
    set_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    data = load_dataset(DATASET, DATASET_CONFIG, split="train")
    train_rows = data.filter(lambda row: row["split"] == "train")
    val_rows = data.filter(lambda row: row["split"] == "validation")
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:   # padded positions are masked in the loss anyway
        tokenizer.pad_token = tokenizer.eos_token

    def encode(row):
        """Chat format of the base model: user = instruction, assistant = code. Loss only on the code tokens."""
        prompt = [{"role": "user", "content": row["instruction"]}]
        full = prompt + [{"role": "assistant", "content": row["code"]}]
        prompt_text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        full_text = tokenizer.apply_chat_template(full, tokenize=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:MAX_LENGTH]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels[:len(full_ids)]}

    columns = train_rows.column_names
    train_ds, val_ds = train_rows.map(encode, remove_columns=columns), val_rows.map(encode, remove_columns=columns)
    # the prompt must be a prefix of the full text, otherwise the loss mask is wrong (checked once per chat template)
    row = train_rows[0]
    prompt = [{"role": "user", "content": row["instruction"]}]
    prompt_text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    full_text = tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content": row["code"]}], tokenize=False)
    assert full_text.startswith(prompt_text), f"chat template: prompt is not a prefix\n{prompt_text!r}\n{full_text!r}"
    print("train:", len(train_ds), "| validation:", len(val_ds))

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                             target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    out_dir = f"outputs/{key}-lora-seed-{SEED}"
    args = TrainingArguments(
        output_dir=out_dir, seed=SEED, data_seed=SEED,
        num_train_epochs=EPOCHS, learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_BATCH, per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM, optim=OPTIMIZER, bf16=True, gradient_checkpointing=True,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=10, report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
    trainer.train()

    adapter_dir = f"{out_dir}/adapter"
    trainer.model.save_pretrained(adapter_dir)
    metrics = {
        "method": "lora", "seed": SEED, "base_model": base,
        "train_samples": len(train_ds), "validation_samples": len(val_ds),
        "best_eval_loss": trainer.state.best_metric, "epochs_run": trainer.state.epoch,
        "train_minutes": round((time.time() - start) / 60, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(f"{adapter_dir}/metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    api.upload_folder(repo_id=repo, folder_path=adapter_dir, path_in_repo=f"seed-{SEED}",
                      commit_message=f"lora adapter {key} seed {SEED}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        repo = ARMS[f"{sys.argv[1]}-lora"]["adapter"]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:   # the machine may stop itself afterwards, so keep the traceback on the Hub
            HfApi().upload_file(path_or_fileobj=error.encode(), path_in_repo=f"errors/{stamp}.txt", repo_id=repo,
                                commit_message=f"training error {stamp}")
        except Exception as upload_exc:
            print("could not upload the error log:", upload_exc, flush=True)
        sys.exit(1)
