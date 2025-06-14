import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from datasets import Dataset
import json
import warnings

# 1. CONFIGURATION - ADJUSTED FOR YOUR HARDWARE
MODEL_NAME = "google/gemma-2b-it"
DATA_PATH = "train_data.json"

# 2. VERIFY DATASET
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {os.path.abspath(DATA_PATH)}")

# 3. MEMORY-EFFICIENT MODEL LOADING
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,  # Reduced precision
    attn_implementation="sdpa"  # More memory efficient
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 4. DATASET PREPARATION WITH STREAMING
def format_instruction(example):
    return {
        "text": f"<start_of_turn>user\n{example['instruction']}<end_of_turn>\n<start_of_turn>model\n{example['response']}<end_of_turn>"
    }

try:
    with open(DATA_PATH) as f:
        raw_data = json.load(f)
    
    # Process in smaller chunks
    chunk_size = 10  # Process 10 examples at a time
    processed_data = []
    for i in range(0, len(raw_data), chunk_size):
        chunk = raw_data[i:i + chunk_size]
        processed_data.extend([format_instruction(d) for d in chunk])
    
    dataset = Dataset.from_list(processed_data)
except Exception as e:
    raise RuntimeError(f"Error loading dataset: {str(e)}")

# 5. MEMORY-EFFICIENT TOKENIZATION
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,  # Reduced from 512
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=4)

# 6. DATA COLLATOR
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 7. PEFT CONFIGURATION
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=4,  # Reduced from 8
    lora_alpha=16,  # Reduced from 32
    target_modules=["q_proj", "v_proj"],  # Fewer target modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# 8. HARDWARE-OPTIMIZED TRAINING ARGS
training_args = TrainingArguments(
    output_dir="./gemma-finetuned",
    per_device_train_batch_size=1,  # Must be 1 for your GPU
    gradient_accumulation_steps=8,  # Increased to compensate
    learning_rate=1e-4,  # Lower learning rate
    num_train_epochs=2,  # Reduced epochs
    fp16=True,
    logging_steps=5,
    optim="paged_adamw_8bit",
    save_strategy="steps",
    save_steps=50,
    evaluation_strategy="no",
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=True,
    max_grad_norm=0.3  # Gradient clipping
)

# 9. TRAINER SETUP
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 10. TRAIN WITH MEMORY MONITORING
print("Starting training...")
print(f"Available GPU Memory: {torch.cuda.mem_get_info()[1]/1024**2:.2f} MB")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    trainer.train()

print("Training completed!")

# 11. SAVE MODEL
model.save_pretrained("./gemma-finetuned")
tokenizer.save_pretrained("./gemma-finetuned")
print("Model saved successfully!")