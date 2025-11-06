from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import torch, json

if __name__ == "__main__":
    model_path = "./qlora-finetuned"
    
    # Load tokenizer + model on GPU
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only = True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        local_files_only = True
    )

    # Load Test Dataset
    dataset = load_dataset("json", data_files = "test_processed.jsonl", split = "train")
    
    def preprocess(example):
        return {"prompt": "[INST] <<SYS>>\nAnswer the question based on the following context: {}\n<</SYS>>\n{}\n[/INST]\n".format(example["context"], example["question"])}
    
    dataset = dataset.map(preprocess, remove_columns = ["context", "question"])

    # Run inference
    for i, item in enumerate(dataset):
        print("=" * 80, flush=True)
        print("Prompt:", item["prompt"], flush=True)

        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p = 0.9,
            repetition_penalty = 1.25,
            num_return_sequences = 1,
            eos_token_id = tokenizer.eos_token_id,
            do_sample=True,
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Generated Answer:", answer[len(item["prompt"]):], flush=True)
        print("\nTrue Answer:", item["answer"], flush = True)
        print("=" * 80, flush=True)

        if i == 5:
            break
