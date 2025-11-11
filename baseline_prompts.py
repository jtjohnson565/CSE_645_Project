import torch, json, re
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.evaluation import load_evaluator
from langchain_community.chat_models import ChatOpenAI

if __name__ == "__main__":
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
    
    eval_metrics = {"correctness": "Is the submission correct, accurate, and factual, assuming that the reference is correct?",
                    "conciseness": "Is the submission concise and to the point, assuming that the reference is concise?",
                    "relevance": "Is the submission referring to a real quote from the text, assuming that the reference is relevant?",
                    "coherence": "Is the submission coherent, well-structured, and organized, assuming that the reference is coherent?",
                    "helpfulness": "Is the submission helpful, insightful, and appropriate, assuming that the reference is helpful"}
    metric_evals = []
    eval_llm = ChatOpenAI(model_name = eval_model_name, temperature = 1.0)

    for m in eval_metrics:
        metric_evals.append(load_evaluator("labeled_score_string", criteria = {m: eval_metrics[m]}, normalize_by = 10, llm = eval_llm))
    
    for i, item in enumerate(dataset):
        print(item["prompt"], end = " ")

        outputs = pl(item["prompt"],
                     do_sample = True,
                     return_full_text = False,
                     temperature = 0.7,
                     top_p = 0.9,
                     repetition_penalty = 1.25,
                     num_return_sequences = 1,
                     max_new_tokens = 200,
                     eos_token_id = tokenizer.eos_token_id)
        
        answer = outputs[0]['generated_text']

        print("Generated Answer:", answer, end = "\n\n")
        print("\nProvided Answer:", item["answer"], end = "\n\n")
        print("\nLangChain Evaluation Metrics:")
        
        for m, ev in zip(eval_metrics.keys(), metric_evals):
            val = ev.evaluate_strings(prediction = answer, reference = item["answer"], input = item["prompt"])
            print("{}: {}".format(m, val['score']))
            print("Reasoning:\n{}\n".format(re.search("^.*(?=\n\nRating: .{5}$)", val['reasoning'], re.M | re.S).group(0)))
        print("\n\n\n")

        if i == 5:
            break
    
    
    avg_metric_vals = {m: 0 for m in eval_metrics}
    num_prompts = 0
    
    for i, item in enumerate(dataset):
        outputs = pl(item["prompt"],
                     do_sample = True,
                     return_full_text = False,
                     temperature = 0.7,
                     top_p = 0.9,
                     repetition_penalty = 1.25,
                     num_return_sequences = 1,
                     max_new_tokens = 200,
                     eos_token_id = tokenizer.eos_token_id)

        answer = outputs[0]['generated_text']
    
        for m, ev in zip(eval_metrics.keys(), metric_evals):
            val = ev.evaluate_strings(prediction = answer, reference = item["answer"], input = item["prompt"])
    
            if val['score'] is not None:
                avg_metric_vals[m] += val['score']
    
        num_prompts += 1
    
        if i == 19:
            break
    
    avg_metric_vals = {m: avg_metric_vals[m] / num_prompts for m in eval_metrics.keys()}
    
    print("Number of Prompts:", num_prompts)
    
    for m in eval_metrics.keys():
        print("Average {} Score: {}".format(m, np.round(avg_metric_vals[m], 3)))
    
