import torch
from transformers import pipeline

model_path="C:\\codes\\gemma22b\\gemma-2-2b-it"



pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",  # replace with "mps" to run on a Mac device
)

question = "आप कैसे हैं"

messages = [
    {"role": "user", "content": question},
]

outputs = pipe(messages, max_new_tokens=512)
assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
print(assistant_response)

# import torch
# from transformers import pipeline

# model_path = "C:\\codes\\gemma22b\\gemma-2-2b-it"

# pipe = pipeline(
#     "text-generation",
#     model=model_path,  
#     model_kwargs={"torch_dtype": torch.float16},  # Use float16 to save VRAM
#     device=0,  
# )

# # Better prompt for translation
# question = "Kese ho bhai"

# # Generate response
# outputs = pipe(question, max_new_tokens=50, return_full_text=False)

# # Extract and print response
# assistant_response = outputs[0]["generated_text"]
# print(assistant_response)
