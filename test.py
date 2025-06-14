import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1. Load with proper configuration
model_id = "tejeshkumarg/ubs"  # Your fine-tuned model

try:
    # Try loading with 4-bit quantization if available
    finbot = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        model_kwargs={
            "load_in_4bit": True,  # Reduces memory usage
            "attn_implementation": "sdpa"  # Better memory efficiency
        }
    )
except Exception as e:
    print(f"Error loading with 4-bit: {str(e)}. Trying normal loading...")
    # Fallback to regular loading
    finbot = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )

# 2. Proper prompt formatting for Gemma
def format_prompt(question):
    return f"""<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
"""

# 3. Generate response with better parameters
question = "एक उदाहरण सहित सरल शब्दों में चक्रवृद्धि ब्याज की व्याख्या कीजिए।"
try:
    response = finbot(
        format_prompt(question),
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=finbot.tokenizer.eos_token_id
    )
    print("Generated Response:")
    print(response[0]['generated_text'])
except Exception as e:
    print(f"Generation error: {str(e)}")
    print("Trying CPU fallback...")
    finbot.model = finbot.model.to('cpu')
    response = finbot(
        format_prompt(question),
        max_new_tokens=128,  # Reduced for CPU
        temperature=0.7
    )
    print(response[0]['generated_text'])