from transformers import pipeline

access_token = "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"
model_id = "google/gemma-2-2b-it"

pipe = pipeline("text-generation", model=model_id, token=access_token)

question = "How are you ?"

messages = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]

outputs = pipe(messages, max_new_tokens=512)
assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
print(assistant_response)
