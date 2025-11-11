import torch
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset

if __name__ == "__main__":
    # ---------------------------
    # Load model & tokenizer
    # ---------------------------
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Requires access from Meta + Hugging Face
    
    model, tokenizer =  FastLanguageModel.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit = True,
            max_seq_length = 2048)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ---------------------------
    # Load and preprocess dataset
    # ---------------------------
    train_dataset = load_dataset("json", data_files = "json_datasets/train_preprocessed.jsonl", split = "train")
    test_dataset = load_dataset("json", data_files = "json_datasets/test_preprocessed.jsonl", split = "train")

    def preprocess(example):
        return {"text": "[INST] <<SYS>>\nYou are a helpful AI assistant that answers biomedical research questions concisely.\n<</SYS>>\nContext:\n{}\nQuestion:\n{}\n[/INST]\n{}".format(example["context"],
                                                                                                                                                                                          example["question"],
                                                                                                                                                                                          example["answer"])}
    
    # Convert Dataset into Conversational Format for SFT Trainer
    train_dataset = train_dataset.map(preprocess, remove_columns = ["context", "question", "answer"])
    test_dataset = test_dataset.map(preprocess, remove_columns = ["context", "question", "answer"])

    
    # ---------------------------
    # LoRA Config
    # ---------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0.2,
        bias = "none",
        random_state = 32
    )

    # ---------------------------
    # Training Arguments
    # ---------------------------
    training_args = TrainingArguments(
        save_strategy = "epoch",
        logging_strategy = "epoch",
        eval_strategy = "epoch",
        output_dir = "./qlora-output",
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        gradient_accumulation_steps = 2,
        learning_rate = 1e-5,
        weight_decay = 0.01,
        warmup_ratio = 0.05,
        lr_scheduler_type = "cosine",
        num_train_epochs = 3,
        report_to="none",
        eval_on_start = True
    )
    
    # ---------------------------
    # Trainer
    # ---------------------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length = 2048
    )


     
    # ---------------------------
    # Train & Save
    # ---------------------------
    trainer.train()
    
    print(" Training complete! Model checkpoints saved to ./qlora-output")
    
