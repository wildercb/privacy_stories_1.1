import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer

# Step 1: Install Dependencies (if not already installed)
# Ensure you have the necessary packages installed
# You may need to run these pip install commands separately
# !pip install "unsloth[colab-new]l tr @ git+https://github.com/unsloth.git"
# !pip instalansformers datasets trl accelerate bitsandbytes

# Step 2: Load Local Dataset
# Read the CSV file
df = pd.read_csv('privacy-analysis-dpo.csv')

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split into train and test sets
dataset_dict = dataset.train_test_split(test_size=0.2)

# If you want to see the actual rows in train and test
train_df = dataset_dict['train'].to_pandas()
test_df = dataset_dict['test'].to_pandas()

# Save these to CSV for reference
train_df.to_csv('train_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

# Optional: Print a few rows to verify
print("\nFirst few rows of training dataset:")
print(train_df.head())
print("\nFirst few rows of test dataset:")
print(test_df.head())

# Step 3: Load Llama 3 Model
max_seq_length = 4096
dtype = None 
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8B-instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Step 4: Prepare LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,  # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Step 5: Patch DPO Trainer (if in notebook)
# PatchDPOTrainer()  # Uncomment if using in a notebook

# Step 6: Configure DPO Training
dpo_trainer = DPOTrainer(
    model = model,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 3,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        learning_rate = 5e-6,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "privacy-dpo-outputs",
    ),
    beta = 0.1,  # Controls the importance of the preference alignment
    train_dataset = dataset_dict["train"],
    eval_dataset = dataset_dict["test"],
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)

# Step 7: Start Training
try:
    dpo_trainer.train()
except Exception as e:
    print(f"Training error: {e}")
    # Handle potential out-of-memory errors by reducing batch size or using smaller model

# Step 8: Save the Model
model.save_pretrained("privacy-dpo-lora-model")

# Step 9: Inference Example
FastLanguageModel.for_inference(model)

# Optional: Create a simple inference pipeline
import transformers

def generate_response(prompt, max_tokens=200):
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
    
    sequences = pipeline(
        formatted_prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=terminators,
        max_length=max_tokens,
        num_return_sequences=1,
    )
    
    return sequences[0]['generated_text'][len(formatted_prompt):].strip()

# Example usage
print(generate_response("Explain privacy considerations for data collection."))
