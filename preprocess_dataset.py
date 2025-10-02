from datasets import load_dataset

# Load PubMedQA
dataset = load_dataset("llamafactory/PubMedQA")

def preprocess(example):
    return {
        "instruction": f"{example['instruction']} {example['input']}",
        "response": example['output']
    }

train_data = dataset["train"].map(preprocess)
train_data.to_json("train_processed.jsonl", orient="records", lines=True)

print(" Preprocessing complete. Saved to train_processed.jsonl")

