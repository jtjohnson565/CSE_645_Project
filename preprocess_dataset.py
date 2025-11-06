from datasets import load_dataset

if __name__ == "__main__":
    # Load PubMedQA
    train_dataset = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split = "train[:85%]")
    test_dataset = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split = "train[-15%:]")
    sft_dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split = "train")
    
    def preprocess(example, has_label = False):
        data_dict = {"context": ' '.join(example['context']['contexts']),
                     "question": example['question'],
                     "answer": example['long_answer']}
        
        if has_label == True:
            data_dict["label"] = example['final_decision']
    
        return data_dict

    train_dataset = train_dataset.map(preprocess, remove_columns = ['pubid', 'long_answer'])
    test_dataset = test_dataset.map(preprocess, remove_columns = ['pubid', 'long_answer'])
    sft_dataset = sft_dataset.map(lambda example: preprocess(example, has_label = True), remove_columns = ['pubid', 'long_answer'])
    
    train_dataset.to_json("train_processed.jsonl", orient = "records", lines = True)
    test_dataset.to_json("test_processed.jsonl", orient = "records", lines = True)
    sft_dataset.to_json("sft_processed.jsonl", orient = "records", lines = True)
    
    print("Preprocessing complete. Saved to train_processed.jsonl, test_processed.jsonl, and sft_processed.jsonl")
