import torch, argparse
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from unsloth_zoo.hf_utils import dtype_from_config

if __name__ == "__main__":
    # Configure Command Line Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int)
    parser.add_argument("--r", type = int)
    parser.add_argument("--alpha", type = float)
    parser.add_argument("--inject_start", type = int, default = 0) # Block to start injecting QLoRA injectors (all if 0, top_n blocks otherwise)
    parser.add_argument("--output_path", type = str)
    args = parser.parse_args()

    print("Output Path:", args.output_path)
    print("Number of Epochs:", args.epochs)
    print("QLoRA Rank:", args.r)
    print("QLoRA Alpha:", args.alpha)
    print("Injection Block Start Point:", args.inject_start, "\n")

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
    
    # Inject Only Top N Layers to Mitigate Catastrophic Forgetting if Argument is Specified to be a Value Greater than 0
    if args.inject_start > 0:
        target_mods = []

        if args.inject_start > 31:
            raise ValueError("Can't Inject More than Total Blocks in Model (32)")

        for i in range(args.inject_start, 32):
            for l in [".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj", ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj"]:
                target_mods.append("layers." + str(i) + l)
        
    else:
        target_mods = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Define PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.r,
        lora_alpha = args.alpha,
        target_modules = target_mods,
        lora_dropout = 0.15,
        bias = "none",
        random_state = 32
    )

    # Print Names of All Layers (Debugging)
    print("Layers:")
    for n, p in model.named_parameters():
        print("\t", n)

    # Verify all layers except for QLoRA injectors are frozen
    print("\nTotal Number of Layers:", len([n for n, p in model.named_parameters()]))
    print("Amount of QLoRA Layers Frozen:", len([n for n, p in model.named_parameters() if "lora" in n and p.requires_grad == False]))
    print("Amount of QLoRA Layers Not Frozen:", len([n for n, p in model.named_parameters() if "lora" in n and p.requires_grad == True]))
    print("Amount of Non-QLoRA Layers Frozen:", len([n for n, p in model.named_parameters() if "lora" not in n and p.requires_grad == False]))
    print("Amount of Non-QLoRA Layers Not Frozen:", len([n for n, p in model.named_parameters() if "lora" not in n and p.requires_grad == True]))
    
    # ---------------------------
    # Training Arguments
    # ---------------------------
    training_args = TrainingArguments(
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
    
    print(" Training complete! Model checkpoints saved to {}".format(args.output_path))
    
