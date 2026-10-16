import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# ================= 配置参数 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # src/lora
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))      # 项目根

# 用 os.path.join 拼绝对路径，避免转义和相对路径问题
MODEL_NAME = os.path.join(
    PROJECT_ROOT, "src", "models", "Qwen", "Qwen2.5-1.5B-Instruct"
)
DATA_PATH = os.path.join(BASE_DIR, "loraData.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "qwen_lora_output")
MAX_LENGTH = 2048


def load_data(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    print("MODEL_NAME =", MODEL_NAME)
    print("DATA_PATH  =", DATA_PATH)
    print("模型存在:", os.path.exists(os.path.join(MODEL_NAME, "configs.json")))
    print("数据存在:", os.path.exists(DATA_PATH))

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 数据
    def preprocess_function(example):
        user_content = example.get("instruction", "")
        if example.get("input"):
            user_content += "\n" + example["input"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False)
        encodings = tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, padding=False
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": list(encodings["input_ids"]),
        }

    raw_data = load_data(DATA_PATH)
    print("加载条数:", len(raw_data))

    dataset = Dataset.from_list(raw_data)
    processed_dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        # num_proc=4,   # Windows 上删掉这行
    )

    # 3. 模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 4. LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 训练
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"训练完成，LoRA 适配器已保存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()