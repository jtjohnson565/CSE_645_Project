import torch, json, re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.evaluation import load_evaluator
from langchain.chat_models import ChatOpenAI

model_name = "meta-llama/Llama-2-7b-hf"
eval_model_name = "gpt-5"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pl = pipeline("text-generation",
              model = model_name,
              dtype = torch.float16,
              tokenizer = tokenizer,
              device_map = "auto")

prompts = []
provided_answers = []

with open("train_processed.jsonl") as f:
    for i, l in enumerate(f):
        data = json.loads(l)

        prompts.append("[INST] <<SYS>>\n{}\n<</SYS>>\n{}\n[/INST]\nAnswer:".format(data['instruction'], data['input']))
        provided_answers.append(data['response'])

        if i == 2:
            break


eval_metrics = ["conciseness", "relevance", "coherence", "helpfulness"]
metric_evals = []
eval_llm = ChatOpenAI(model_name = eval_model_name, temperature = 1.0)

for m in eval_metrics:
    metric_evals.append(load_evaluator("criteria", criteria = m, llm = eval_llm))

for prompt, ans in zip(prompts, provided_answers):
    print(prompt, end = " ")

    outputs = pl(prompt,
                 do_sample = True,
                 return_full_text = False,
                 temperature = 0.7,
                 top_p = 0.9,
                 repetition_penalty = 1.25,
                 num_return_sequences = 1,
                 max_new_tokens = 200,
                 eos_token_id = tokenizer.eos_token_id)
    
    print(outputs[0]['generated_text'], end = "\n\n")
    print("Provided Answer:", ans, end = "\n\n")
    print("LangChain Evaluation Metrics:")
    
    for m, ev in zip(eval_metrics, metric_evals):
        val = ev.evaluate_strings(prediction = outputs[0]['generated_text'],
                                 input = prompt)
        print("{}: {}".format(m, val['score']))
        print("Reasoning:\n{}\n".format(re.search("^.*(?=\n\n[Y|N]$)", val['reasoning'], re.M | re.S).group(0)))
    print("\n\n\n")


avg_metric_vals = {m: 0 for m in eval_metrics}
num_prompts = 0

with open("train_processed.jsonl") as f:
    for i, l in enumerate(f):
        data = json.loads(l)

        prompt = "[INST] <<SYS>>\n{}\n<</SYS>>\n{}\n[/INST]\nAnswer:".format(data['instruction'], data['input'])

        outputs = pl(prompt,
                     do_sample = True,
                     return_full_text = False,
                     temperature = 0.7,
                     top_p = 0.9,
                     repetition_penalty = 1.25,
                     num_return_sequences = 1,
                     max_new_tokens = 200,
                     eos_token_id = tokenizer.eos_token_id)

        for m, ev in zip(eval_metrics, metric_evals):
            val = ev.evaluate_strings(prediction = outputs[0]['generated_text'],
                                 input = prompt)

            if val['score'] is not None:
                avg_metric_vals[m] += val['score']

        num_prompts += 1

        if i == 99:
            break

avg_metric_vals = {m: avg_metric_vals[m] / num_prompts for m in eval_metrics}

print("Number of Prompts:", num_prompts)

for m in eval_metrics:
    print("Average {} Score: {}".format(m, np.round(avg_metric_vals[m], 3)))
