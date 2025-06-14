import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "google/gemma-2-2b-it"

# Quantization with bitsandbytes (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Load in 4-bit precision
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load Model
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model Loaded Successfully")
