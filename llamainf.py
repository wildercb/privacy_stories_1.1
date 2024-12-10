import os
import torch
import pandas as pd
import transformers
from unsloth import FastLanguageModel

# Load the original dataset to get the prompts
df = pd.read_csv('LLMAnnotation-gpt40-shashank.csv')

# Load the trained model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    "privacy-dpo-lora-model",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

# Prepare the model for inference
model = FastLanguageModel.for_inference(model)

# Create a generation function
def generate_response(prompt, max_new_tokens=200):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in privacy analysis."},
        {"role": "user", "content": prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    try:
        sequences = pipeline(
            formatted_prompt,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=terminators,
            max_new_tokens=max_new_tokens,  # Changed from max_length to max_new_tokens
            num_return_sequences=1,
        )

        return sequences[0]['generated_text'][len(formatted_prompt):].strip()
    except Exception as e:
        print(f"Error generating response for prompt: {prompt}")
        print(f"Error: {e}")
        return "Error generating response"

# Select rows 21-25 (Python uses 0-based indexing, so rows 20-24)
selected_rows = df.iloc[20:26]

# Create a list to store results
results_list = []

# Generate responses for selected prompts
for index, row in selected_rows.iterrows():
    prompt = row['Prompt']  # Assuming the column is named 'Prompt'
    response = generate_response(prompt)

    # Append to results list
    results_list.append({
        'Original Prompt': prompt,
        'Model Response': response,
        'Target Annotations': row['Target Annotations'],  # Include Target Annotations
        'File': row['File']  # Include File
    })

# Create DataFrame from the list
results_df = pd.DataFrame(results_list)

# Save results to CSV
results_df.to_csv('newllama_responses.csv', index=False)
print("Responses generated and saved to newllama_responses.csv")
print("\nGenerated Responses:")
print(results_df)