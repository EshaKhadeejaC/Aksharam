import os
import json
import torch
from datasets import Dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType

def load_aligned_data(file_path):
    print(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples.")
    return Dataset.from_list(data)

def preprocess_function(examples, tokenizer, max_length=128):
    inputs = examples["en"]
    targets = examples["ml"]
    
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
    return model_inputs

def main():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    data_path = "aligned_dataset.json"

    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found. Please run 01_collect_and_align.py first.")
        return

    print("Loading tokenizer...")
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    tokenizer.src_lang = "en_XX"
    # Ensure ml_IN is in the target languages so we can force it natively
    # Some older versions might need manual addition, but mbart-large-50-many-to-many-mmt supports ml_IN

    print("Loading dataset...")
    dataset = load_aligned_data(data_path)
    
    # Split into train and validation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Dramatically reduce size for demonstration purposes
    dataset["train"] = dataset["train"].select(range(min(40, len(dataset["train"]))))
    dataset["test"] = dataset["test"].select(range(min(10, len(dataset["test"]))))
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # We load in standard FP32 or FP16 depending on hardware. BF16/FP16 is recommended for 600M models.
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    print("Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"] # Targeting attention blocks is standard for LoRA on transformers
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./aksharam_model",
        eval_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=8, # Keep batch size small to avoid OOM
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3, # 3 epochs for demonstration
        predict_with_generate=True,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        logging_steps=10,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving the LoRA model...")
    trainer.save_model("./aksharam_model_final")
    
    print("Training complete! Model adapters saved to ./aksharam_model_final")

if __name__ == "__main__":
    main()
