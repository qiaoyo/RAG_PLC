import argparse
import logging
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from data_models import PLCExample
from ingestion import load_plc_dataset
from pipeline import SYSTEM_PROMPT
from settings import DATASET_PATH, DEFAULT_GEN_MODEL, DEFAULT_TOKENIZER_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_training_text(example: PLCExample) -> str:
    input_block = example.input if example.input else "无"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"用户指令: {example.instruction}\n"
        f"补充输入: {input_block}\n"
        f"请提供PLC代码：\n{example.output}"
    )


def prepare_dataset(dataset_path: Path) -> Dataset:
    raw = load_plc_dataset(dataset_path)
    texts: List[str] = [build_training_text(item) for item in raw]
    return Dataset.from_dict({"text": texts})


def finetune(
    model_name: str,
    tokenizer_path: str,
    dataset_path: Path,
    output_dir: Path,
    lr: float = 1e-4,
    epochs: int = 2,
    per_device_batch_size: int = 2,
    max_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    ds = prepare_dataset(dataset_path)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="longest",
            max_length=max_length,
        )

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=[],
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("LoRA adapter saved to %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA finetuning for PLC code generation")
    parser.add_argument("--model", type=str, default=DEFAULT_GEN_MODEL, help="Base model for SFT")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH, help="Tokenizer path")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Path to PLC JSON dataset")
    parser.add_argument("--output", type=Path, default=Path("rag/artifacts/plc_lora"), help="Output directory")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finetune(
        model_name=args.model,
        tokenizer_path=args.tokenizer,
        dataset_path=args.dataset,
        output_dir=args.output,
        lr=args.lr,
        epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
