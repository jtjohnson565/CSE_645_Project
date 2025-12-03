from datasets import load_dataset


def build_context_with_subheadings(contexts, headings=None, max_chunks=None):
    if isinstance(contexts, str):
        return contexts.strip()

    parts = []
    for i, ctx in enumerate(contexts):
        if ctx is None:
            continue
        ctx = str(ctx).strip()
        if not ctx:
            continue

        h = None
        if isinstance(headings, list) and i < len(headings):
            h = str(headings[i]).strip()

        parts.append(f"{h.upper()}: {ctx}" if h else ctx)

    if max_chunks is not None and len(parts) > max_chunks:
        parts = parts[:max_chunks]

    return "\n---\n".join(parts)


def preprocess(example, has_label=False):
    ctx_obj = example["context"]  # {"contexts":[...], "labels":[...]}
    contexts = ctx_obj.get("contexts", [])
    headings = ctx_obj.get("labels") or ctx_obj.get("headings") or ctx_obj.get("subheadings") or None

    ctx_all = build_context_with_subheadings(contexts, headings, max_chunks=None)
    ctx_1 = build_context_with_subheadings(contexts, headings, max_chunks=1)
    ctx_2 = build_context_with_subheadings(contexts, headings, max_chunks=2)

    data_dict = {
        "context": ctx_all,          # backward compatible
        "context_all": ctx_all,
        "context_1": ctx_1,
        "context_2": ctx_2,
        "question": example["question"],
        "answer": example["long_answer"],
    }

    if has_label:
        data_dict["label"] = example["final_decision"]

    return data_dict


if __name__ == "__main__":
    train_dataset = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split="train[:85%]")
    test_dataset = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split="train[-15%:]")
    labeled_dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    train_dataset = train_dataset.map(preprocess, remove_columns=["pubid", "long_answer"])
    test_dataset = test_dataset.map(preprocess, remove_columns=["pubid", "long_answer"])
    labeled_dataset = labeled_dataset.map(
        lambda ex: preprocess(ex, has_label=True),
        remove_columns=["pubid", "long_answer"],
    )

    train_dataset.to_json("json_datasets/train_preprocessed.jsonl", orient="records", lines=True)
    test_dataset.to_json("json_datasets/test_preprocessed.jsonl", orient="records", lines=True)
    labeled_dataset.to_json("json_datasets/labeled_preprocessed.jsonl", orient="records", lines=True)

    print("Preprocessing complete. Saved JSONL files to json_datasets/.")
    print("Number of Rows in Training Dataset:", len(train_dataset))
    print("Number of Rows in Testing Dataset:", len(test_dataset))
    print("Number of Rows in Labeled Dataset:", len(labeled_dataset))
