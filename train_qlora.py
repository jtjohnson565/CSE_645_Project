import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset

# ---------------------------
# Load model & tokenizer
# ---------------------------
model_name = "meta-llama/Llama-2-7b-hf"  # Requires access from Meta + Hugging Face

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True
)

# ---------------------------
# Load and preprocess dataset
# ---------------------------
dataset = load_dataset("llamafactory/PubMedQA")

def preprocess(example):
    text = f"{example['instruction']} {example['input']} {example['output']}"
    return {"text": text}

dataset = dataset.map(preprocess)
train_dataset = dataset["train"]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ---------------------------
# LoRA Config
# ---------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ---------------------------
# Training Arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir="./qlora-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

# ---------------------------
# Trainer
# ---------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    peft_config=lora_config,
    data_collator=data_collator
)

# ---------------------------
# Train & Save
# ---------------------------
trainer.train()
model.save_pretrained("./qlora-finetuned")
tokenizer.save_pretrained("./qlora-finetuned")

print(" Training complete! Model saved to ./qlora-finetuned")


