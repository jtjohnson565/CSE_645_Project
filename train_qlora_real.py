import torch, argparse, os, json
import numpy as np
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, TaskType
from datasets import load_dataset
from unsloth_zoo.hf_utils import dtype_from_config

if __name__ == "__main__":
    # Configure Command Line Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int)
    parser.add_argument("--r", type = int)
    parser.add_argument("--alpha", type = float)
    parser.add_argument("--synthetic_model_id", type = int)
    parser.add_argument("--synthetic_checkpoint_steps", type = int)
    parser.add_argument("--output_path", type = str)
    args = parser.parse_args()

    checkpoint_path = "synthetic_checkpoints/qlora-output-{}/checkpoint-{}".format(args.synthetic_model_id, args.synthetic_checkpoint_steps)

    with open(os.path.join(checkpoint_path, "adapter_config.json"), "r") as f:
        synthetic_qlora_config = json.load(f)

    with open(os.path.join(checkpoint_path, "trainer_state.json"), "r") as f:
        synthetic_trainer_config = json.load(f)

    print("Checkpoint Path:", checkpoint_path)
    print("Output Path:", args.output_path)
    print("Number of Synthetic Epochs:", synthetic_trainer_config["epoch"])
    print("Synthetic QLoRA Rank:", synthetic_qlora_config["r"])
    print("Synthetic QLoRA Alpha:", synthetic_qlora_config["lora_alpha"])
    print("Number of Real Epochs:", args.epochs)
    print("Real QLoRA Rank:", args.r)
    print("Real QLoRA Alpha:", args.alpha, "\n")
    
    # ---------------------------
    # Load model & tokenizer
    # ---------------------------
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Requires access from Meta + Hugging Face

    model, tokenizer =  FastLanguageModel.from_pretrained(
            checkpoint_path,
            device_map="auto",
            load_in_4bit = True,
            max_seq_length = 2048)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------
    # Load and preprocess dataset
    # ---------------------------
    real_train_dataset = load_dataset("json", data_files = "json_datasets/train_preprocessed.jsonl", split = "train")
    val_dataset = load_dataset("json", data_files = "json_datasets/val_preprocessed.jsonl", split = "train")

    def preprocess(example):
        return {"text": "[INST] <<SYS>>\nYou are a helpful AI assistant that answers biomedical research questions concisely.\n<</SYS>>\nContext:\n{}\nQuestion:\n{}\n[/INST]\n{}".format(example["context"],
                                                                                                                                                                                          example["question"],
                                                                                                                                                                                          example["answer"])}
    
    # Convert Dataset into Conversational Format for SFT Trainer
    real_train_dataset = real_train_dataset.map(preprocess, remove_columns = ["context", "question", "answer"])
    val_dataset = val_dataset.map(preprocess, remove_columns = ["context", "question", "answer"])

    # ---------------------------
    # LoRA Config
    # ---------------------------
    real_adapter_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM,
        r = args.r,
        lora_alpha = args.alpha,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0.15,
        bias = "none",
    )

    model.add_adapter(peft_config = real_adapter_config, adapter_name = "real_lora")
    model.set_adapter("real_lora")

    # Print Names of All Layers (Debugging)
    print("Layers:")
    for n, p in model.named_parameters():
        print("\t", n)

    # Verify all layers except for New QLoRA injectors are frozen
    print("\nTotal Number of Layers:", len([n for n, p in model.named_parameters()]))
    print("Amount of New QLoRA Layers Frozen:", len([n for n, p in model.named_parameters() if ".real_lora" in n and p.requires_grad == False]))
    print("Amount of New QLoRA Layers Not Frozen:", len([n for n, p in model.named_parameters() if ".real_lora" and p.requires_grad == True]))
    print("Amount of Old QLoRA Layers Frozen:", len([n for n, p in model.named_parameters() if ".default" in n and p.requires_grad == False]))
    print("Amount of Old QLoRA Layers Not Frozen:", len([n for n, p in model.named_parameters() if ".default" in n and p.requires_grad == True]))
    print("Amount of Non-QLoRA Layers Frozen:", len([n for n, p in model.named_parameters() if "lora" not in n and p.requires_grad == False]))
    print("Amount of Non-QLoRA Layers Not Frozen:", len([n for n, p in model.named_parameters() if "lora" not in n and p.requires_grad == True]))

    # ---------------------------
    # Real Training Arguments
    # ---------------------------
    real_training_args = TrainingArguments(
        save_strategy = "epoch",
        logging_strategy = "epoch",
        eval_strategy = "epoch",
        output_dir = args.output_path,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        gradient_accumulation_steps = 2,
        learning_rate = 1e-5,
        weight_decay = 0.01,
        warmup_ratio = 0.05,
        lr_scheduler_type = "cosine",
        num_train_epochs = args.epochs,
        report_to="none",
        eval_on_start = True
    )

    # ---------------------------
    # Real Trainer
    # ---------------------------
    real_trainer = SFTTrainer(
        model=model,
        train_dataset=real_train_dataset,
        eval_dataset=val_dataset,
        args=real_training_args,
        tokenizer=tokenizer,
        max_seq_length = 2048
    )
     
    # ---------------------------
    # Train Real Data & Save
    # ---------------------------
    real_trainer.train()
    
    print("\nTraining complete! Model checkpoints saved to {}".format(args.output_path))
    
