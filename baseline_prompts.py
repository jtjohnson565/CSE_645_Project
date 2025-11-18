import torch, json, re
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI

if __name__ == "__main__":
    PRINT_PROMPT_LIMIT = 5
    AGGREGATION_LIMIT = 20

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    eval_model_name = "gpt-5"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    pl = pipeline("text-generation",
                  model = model_name,
                  dtype = torch.float16,
                  tokenizer = tokenizer,
                  device_map = "auto")
    
    # Load Test Dataset
    dataset = load_dataset("json", data_files = "json_datasets/test_preprocessed.jsonl", split = "train")

    def preprocess(example):
        return {"prompt": "[INST] <<SYS>>\nYou are a helpful AI assistant that answers biomedical research questions concisely.\n<</SYS>>\nContext:\n{}\nQuestion:\n{}\n[/INST]\n".format(example["context"],
                                                                                                                                                                                          example["question"])}

    dataset = dataset.map(preprocess, remove_columns = ["context", "question"])

    eval_metrics = {"correctness": "Does the response correctly, accurately, and factually answer the question, assuming that the answer provided is correct?",
                    "conciseness": "Irrespective of correctness, does the response concisely answer the question and to the point, based on the answer provided?",
                    "relevance": "Irrespective of correctness, does the response answer the question while referring to a real quote from the context, based on the answer provided?",
                    "coherence": "Irrespective of correctness, does the response answer the question coherently while being well-structured and organized, based on the answer provided?",
                    "helpfulness": "Irrespective of correctness, does the response answer the question helpfully, insightfully, and appropriately, based on the answer provided?"}
    
    metric_evals = []
    eval_llm = ChatOpenAI(model_name = eval_model_name, temperature = 0)

    for m in eval_metrics:
        metric_evals.append(load_evaluator("labeled_score_string", criteria = {m: eval_metrics[m]}, normalize_by = 10, llm = eval_llm))

    print("Maximum number of tokens in answers:", np.max([len(tokenizer(item["answer"])["input_ids"]) for item in dataset]))
    print("Average number of tokens in answers:", np.average([len(tokenizer(item["answer"])["input_ids"]) for item in dataset]))
    print("Evaluation Model Used:", eval_model_name)
    print("Number of Prompts Printed:", PRINT_PROMPT_LIMIT)
    print("Number of Prompts Used for Calculating the Average for Each Metric:", AGGREGATION_LIMIT, "\n")

    avg_metric_vals = {m: 0 for m in eval_metrics}

    for i, item in enumerate(dataset):
        outputs = pl(item["prompt"],
                     do_sample = True,
                     return_full_text = False,
                     temperature = 0.6,
                     top_p = 0.9,
                     num_return_sequences = 1,
                     max_new_tokens = 750,
                     eos_token_id = tokenizer.eos_token_id)

        answer = outputs[0]['generated_text']

        if i < PRINT_PROMPT_LIMIT - 1:
            print("=" * 80, flush=True)
            print("Prompt:", item["prompt"], flush=True)

            print("Generated Answer:", answer, flush=True)
            print("\nProvided Answer:", item["answer"], "\n", flush = True)

        for m, ev in zip(eval_metrics.keys(), metric_evals):
            val = ev.evaluate_strings(prediction = answer, reference = item["answer"], input = item["prompt"])

            if val['score'] is not None:
                avg_metric_vals[m] += val['score']

            if i < PRINT_PROMPT_LIMIT - 1:
                print("{}: {}".format(m, val['score']))

                reasoning = re.search("^.*(?=\n\nRating: .{5,6}$)", val['reasoning'], re.M | re.S)
                print("Reasoning:\n{}\n".format(reasoning.group(0)))

        if i < PRINT_PROMPT_LIMIT - 1:
            print("=" * 80, flush=True)

        if i == AGGREGATION_LIMIT - 1:
            break

    avg_metric_vals = {m: avg_metric_vals[m] / AGGREGATION_LIMIT for m in eval_metrics.keys()}

    for m in eval_metrics.keys():
        print("Average {} Score: {}".format(m, np.round(avg_metric_vals[m], 3)))
