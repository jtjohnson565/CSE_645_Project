import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset

if __name__ == "__main__":
    # ---------------------------
    # Load model & tokenizer
    # ---------------------------
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Requires access from Meta + Hugging Face
    
    quant_config = BitsAndBytesConfig(load_in_4bit = True,
                                      bnb_4bit_quant_type = "nf4",
                                      bnb_4bit_compute_dtype = torch.bfloat16,
                                      bnb_4bit_use_double_quant = True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config = quant_config
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token = True,
                                              max_length = 4096,
                                              padding = "max_length",
                                              truncation = "max_length")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ---------------------------
    # Load and preprocess dataset
    # ---------------------------
    train_dataset = load_dataset("json", data_files = "train_processed.jsonl", split = "train")
    test_dataset = load_dataset("json", data_files = "test_processed.jsonl", split = "train")

    def preprocess(example):   
        return {"text": "[INST] <<SYS>>\nAnswer the question based on the following context: {}\n<</SYS>>\n{}\n[/INST]\n{}".format(example["context"], example["question"], example["answer"])}
    
    # Convert Dataset into Conversational Format for SFT Trainer
    train_dataset = train_dataset.map(preprocess, remove_columns = ["context", "question", "answer"])
    test_dataset = test_dataset.map(preprocess, remove_columns = ["context", "question", "answer"])

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
        save_strategy = "epoch",
        logging_strategy = "epoch",
        eval_strategy = "epoch",
        output_dir = "./qlora-output",
        overwrite_output_dir = True,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        learning_rate = 1e-4,
        num_train_epochs = 10,
        fp16=True,
        report_to="none",

    )
    
    # ---------------------------
    # Trainer
    # ---------------------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        peft_config=lora_config,
        data_collator=data_collator
    )
    
    # ---------------------------
    # Train & Save
    # ---------------------------
    trainer.train()
    trainer.save_model("./qlora-finetuned")
    
    print(" Training complete! Model saved to ./qlora-finetuned")
