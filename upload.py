from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login
import torch
import os

# 1. Set your paths
MODEL_DIR = "./gemma-finetuned"  # Your local model directory
HF_REPO = "tejeshkumarg/ubs"     # Your Hugging Face repo
TOKEN = "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"          # From https://huggingface.co/settings/tokens

# 2. Generate config if missing
if not os.path.exists(f"{MODEL_DIR}/config.json"):
    print("Generating config.json...")
    config = AutoConfig.from_pretrained("google/gemma-2b-it")  # Original base model
    
    # If you modified architecture during fine-tuning, update config here
    # config.your_custom_setting = value  
    
    config.save_pretrained(MODEL_DIR)

# 3. Verify all required files exist
required_files = ["config.json", "adapter_model.safetensors"] + \
                 [f for f in os.listdir(MODEL_DIR) if f.startswith("tokenizer")]
assert all(os.path.exists(f"{MODEL_DIR}/{f}") for f in required_files), \
       "Missing essential files!"

# 4. Upload with verification
login(token=TOKEN)
api = HfApi()

api.upload_folder(
    folder_path=MODEL_DIR,
    repo_id=HF_REPO,
    repo_type="model",
    commit_message="Adding missing config.json",
    allow_patterns=["*.json", "*.model", "*.safetensors", "*.py"]
)

print(f"Upload complete! Verify at: https://huggingface.co/{HF_REPO}")