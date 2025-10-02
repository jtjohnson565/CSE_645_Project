import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "meta-llama/Llama-2-7b-hf"

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

for prompt, ans in zip(prompts, provided_answers):
    print(prompt, end = " ")

    outputs = pl(prompt,
                 do_sample = True,
                 return_full_text = False,
                 temperature = 1.2,
                 top_k = 50,
                 top_p = 0.95,
                 num_beams = 2,
                 repetition_penalty = 1.75,
                 length_penalty = 1.2,
                 num_return_sequences = 1,
                 max_new_tokens = 200,
                 eos_token_id = tokenizer.eos_token_id)
    
    print(outputs[0]['generated_text'], end = "\n\n")
    print("Provided Answer:", ans, end = "\n\n\n\n")
