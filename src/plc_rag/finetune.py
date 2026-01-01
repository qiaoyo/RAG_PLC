"""LoRA fine-tuning utilities for PLC instruction-following."""

from __future__ import annotations

import logging
from functools import partial
from typing import Dict

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .config import LoraFineTuneConfig

LOGGER = logging.getLogger(__name__)

PROMPT_TEMPLATE = """<|system|>
你是一位擅长PLC控制与结构化文本(ST)语言的高级工程师，输出需要满足IEC 61131-3规范。
</s>
<|user|>
{instruction}

{inputs}
</s>
<|assistant|>
{output}
</s>
"""


def _format_prompt(record: Dict) -> str:
    instruction = record.get("instruction", "")
    inputs = record.get("input", "")
    output = record.get("output", "")
    prepared_inputs = inputs if inputs else "无额外输入。"
    return PROMPT_TEMPLATE.format(
        instruction=instruction.strip(),
        inputs=prepared_inputs.strip(),
        output=output.strip(),
    )


def _tokenize_function(tokenizer, max_seq_length: int, batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )


def run_lora_finetune(config: LoraFineTuneConfig) -> None:
    """Run supervised fine-tuning with LoRA adapters."""

    LOGGER.info("Loading dataset from %s", config.dataset_path)
    dataset = load_dataset("json", data_files=str(config.dataset_path))["train"]
    dataset = dataset.shuffle(seed=config.seed)
    dataset = dataset.map(lambda r: {"text": _format_prompt(r)})

    split = dataset.train_test_split(test_size=0.05, seed=config.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    LOGGER.info("Loading tokenizer/model %s", config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    tokenize_fn = partial(_tokenize_function, tokenizer, config.max_seq_length)
    tokenized_train = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_eval = eval_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.micro_batch_size,
        per_device_eval_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.save_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    LOGGER.info("LoRA fine-tuned model saved to %s", config.output_dir)
