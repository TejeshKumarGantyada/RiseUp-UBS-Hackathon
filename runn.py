from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 1. Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# 2. Load base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder="offload"
)

# 3. Load adapters
model = PeftModel.from_pretrained(base_model, "tejeshkumarg/ubs")

# 4. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

def generate_response(question, max_new_tokens=200):
    # Format prompt with Gemma's special tokens
    prompt = f"""<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
"""
    # Tokenize and move to model's device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # Decode and clean output
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return full_response.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()

# Example usage
print(generate_response("Explain compound interest with an example"))