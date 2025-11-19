from datasets import load_dataset

if __name__ == "__main__":
    # Load PubMedQA
    real_train_dataset = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split = "train[:80%]")
    synthetic_train_dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split = "train")
    val_dataset = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split = "train[80%:90%]")
    test_dataset = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split = "train[90%:]")
    labeled_dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split = "train")
    
    def preprocess(example, has_label = False):
        data_dict = {"context": ' '.join(example['context']['contexts']),
                     "question": example['question'],
                     "answer": example['long_answer']}
        
        if has_label == True:
            data_dict["label"] = example['final_decision']
    
        return data_dict

    real_train_dataset = real_train_dataset.map(preprocess, remove_columns = ['pubid', 'long_answer'])
    synthetic_train_dataset = synthetic_train_dataset.map(preprocess, remove_columns = ['pubid', 'long_answer', 'final_decision'])
    val_dataset = val_dataset.map(preprocess, remove_columns = ['pubid', 'long_answer'])
    test_dataset = test_dataset.map(preprocess, remove_columns = ['pubid', 'long_answer'])
    labeled_dataset = labeled_dataset.map(lambda example: preprocess(example, has_label = True), remove_columns = ['pubid', 'long_answer'])
    
    real_train_dataset.to_json("json_datasets/train_preprocessed.jsonl", orient = "records", lines = True)
    synthetic_train_dataset.to_json("json_datasets/train_synthetic_preprocessed.jsonl", orient = "records", lines = True)
    val_dataset.to_json("json_datasets/val_preprocessed.jsonl", orient = "records", lines = True)
    test_dataset.to_json("json_datasets/test_preprocessed.jsonl", orient = "records", lines = True)
    labeled_dataset.to_json("json_datasets/labeled_preprocessed.jsonl", orient = "records", lines = True)

    print("Number of Rows in Real Training Dataset:", len(real_train_dataset))
    print("Number of Rows in Synthetic Training Dataset:", len(synthetic_train_dataset))
    print("Number of Rows in Validation Dataset:", len(val_dataset))
    print("Number of Rows in Testing Dataset:", len(test_dataset))
    print("Number of Rows in Labeled Dataset:", len(labeled_dataset))

    print("Preprocessing complete. Saved to json_datasets directory.")
