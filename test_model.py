from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Path to your fine-tuned model
model_path = "./qlora-finetuned"

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load tokenizer + model on GPU
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

# Example prompts
prompts = [
    "Question: Is naturopathy as effective as conventional therapy for treatment of menopausal symptoms?",
    "Question: Can randomized trials rely on existing electronic data?",
    "Question: Does exercise improve sleep quality in adults?",
]

# Run inference
for prompt in prompts:
    print("=" * 80, flush=True)
    print("Prompt:", prompt, flush=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Generated Answer:", answer, flush=True)
    print("=" * 80, flush=True)

