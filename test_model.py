from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import load_dataset
from langchain.evaluation import load_evaluator
from langchain_community.chat_models import ChatOpenAI
import torch, json, re, argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type = int)
    parser.add_argument("--checkpoint_step", type = int)
    parser.add_argument("--eval_model", type = str)
    args = parser.parse_args()

    PRINT_PROMPT_LIMIT = 5
    AGGREGATION_LIMIT = 20
    
    model_checkpoint_path = "./qlora-output-{}/checkpoint-{}".format(args.model_id, args.checkpoint_step)
    
    eval_model_name = args.eval_model  # Used gpt-5-mini for hyperparameter tuning and gpt-5 for final evaluation.
                                    
    # Load tokenizer + model on GPU
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path, local_files_only = True)
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint_path,
        device_map="auto",
        local_files_only = True
    )

    pl = pipeline("text-generation",
                  model = model,
                  dtype = torch.float16,
                  tokenizer = tokenizer,
                  device_map = "auto")

    # Load Test Dataset
    dataset = load_dataset("json", data_files = "json_datasets/test_preprocessed.jsonl", split = "train")

    def preprocess(example):
        return {"prompt": "[INST] <<SYS>>\nYou are a helpful AI assistant that answers biomedical research questions concisely.\n<</SYS>>\nContext:\n{}\nQuestion:\n{}\n[/INST]\n".format(example["context"],
                                                                                                                                                                                          example["question"])}
    
    dataset = dataset.map(preprocess, remove_columns = ["context", "question"])
    
    # Define Evaluation Metrics and Criteria
    eval_metrics = {"correctness": "Does the response correctly, accurately, and factually answer the question, assuming that the answer provided is perfectly correct?",
                    "conciseness": "Irrespective of correctness, does the response concisely answer the question and to the point, based on the answer provided?",
                    "relevance": "Irrespective of correctness, does the response answer the question while referring to a real quote from the context, based on the answer provided?",
                    "coherence": "Irrespective of correctness, does the response answer the question coherently while being well-structured and organized, based on the answer provided?",
                    "helpfulness": "Irrespective of correctness, does the response answer the question helpfully, insightfully, and appropriately, based on the answer provided?"}
    metric_evals = []
    eval_llm = ChatOpenAI(model_name = eval_model_name, temperature = 1.0)

    for m in eval_metrics.keys():
        metric_evals.append(load_evaluator("labeled_score_string", criteria = {m: eval_metrics[m]}, normalize_by = 10, llm = eval_llm))

    print("Model Checkpoint Epoch:", args.checkpoint_step / 1627)
    print("Evaluation Model Used:", eval_model_name)
    print("Number of Prompts Printed:", PRINT_PROMPT_LIMIT)
    print("Number of Prompts Used for Calculating the Average for Each Metric:", AGGREGATION_LIMIT, "\n")

    model.eval()
    
    avg_metric_vals = {m: 0 for m in eval_metrics}

    for i, item in enumerate(dataset):
        outputs = pl(item["prompt"],
                     do_sample = True,
                     return_full_text = False,
                     temperature = 0.6,
                     top_p = 0.9,
                     repetition_penalty = 1.1,
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

                reasonings = [re.search("^.*(?={}$)".format(r), val['reasoning'], re.M | re.S) for r in ["\n\nRating: .{5,6}", "Rating: .{5,6}", ".{5,6}"]]

                for j, r in enumerate(reasonings):
                    if r is not None:
                        print("Reasoning:\n{}\n".format(r.group(0)))
                        break

                    if j == 2:
                        print("No Reasoning Provided\n")

        if i < PRINT_PROMPT_LIMIT - 1:
            print("=" * 80, flush=True)

        if i == AGGREGATION_LIMIT - 1:
            break

    avg_metric_vals = {m: avg_metric_vals[m] / AGGREGATION_LIMIT for m in eval_metrics.keys()}

    for m in eval_metrics.keys():
        print("Average {} Score: {}".format(m, np.round(avg_metric_vals[m], 3)))
    
