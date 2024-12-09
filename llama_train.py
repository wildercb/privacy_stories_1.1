import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments
)
from trl import DPOTrainer
import bitsandbytes as bnb
from huggingface_hub import HfApi, HfFolder
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)

def load_and_prepare_dataset(csv_path, max_samples=None):
    """
    Load and optimize dataset for faster processing
    
    Args:
        csv_path (str): Path to RLHF CSV
        max_samples (int, optional): Limit number of training samples
    
    Returns:
        datasets.Dataset: Optimized dataset
    """
    df = pd.read_csv(csv_path, engine='c', low_memory=True)
    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)))
    dataset = Dataset.from_pandas(df[['Prompt', 'Model Response 1', 'Model Response 2', 'Preferred_Response']])

    def format_dataset(samples):
        preference = int(samples['Preferred_Response'])
        return {
            'prompt': samples['Prompt'],
            'chosen': samples[f'Model Response {preference}'],
            'rejected': samples[f'Model Response {3 - preference}']
        }

    return dataset.map(format_dataset, remove_columns=dataset.column_names, batched=True, batch_size=1000)

def prepare_model_and_tokenizer(model_name):
    """
    Prepare quantized and memory-efficient model
    
    Args:
        model_name (str): Hugging Face model identifier
    
    Returns:
        tuple: (quantized model, tokenizer)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map='auto',
        load_in_4bit=True,
        quantization_config=bnb.LLM.prepare_model_for_kbit_training(),
        torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,  # Rank of LoRA adaptation
        lora_alpha=32,  # Scaling factor
        target_modules=['q_proj', 'v_proj'],  # Key transformer modules
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True,
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def configure_training_args(output_dir):
    """
    Configure training arguments
    
    Args:
        output_dir (str): Training output directory
    
    Returns:
        TrainingArguments: Configured arguments
    """
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=True,
        bf16=True,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        save_strategy='epoch',
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        ddp_find_unused_parameters=False
    )

def train_rlhf_model(
    csv_path, 
    model_name='meta-llama/Llama-3.2-3b-instruct', 
    output_dir='./llama_rlhf_output',
    max_samples=5000
):
    """
    Train RLHF model and upload to Hugging Face Hub
    
    Args:
        csv_path (str): Path to RLHF dataset
        model_name (str): Model identifier
        output_dir (str): Output directory
        max_samples (int): Maximum training samples
    """
    os.makedirs(output_dir, exist_ok=True)
    formatted_dataset = load_and_prepare_dataset(csv_path, max_samples)
    model, tokenizer = prepare_model_and_tokenizer(model_name)
    training_args = configure_training_args(output_dir)

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
    )
    torch.cuda.empty_cache()

    print("🚀 Initiating High-Performance RLHF Training...")
    dpo_trainer.train()

    print("✅ Training Complete. Uploading to Hugging Face Hub...")
    model.push_to_hub(output_dir)
    tokenizer.push_to_hub(output_dir)

    print(f"✅ Model and tokenizer uploaded to Hugging Face Hub.")

def main():
    train_rlhf_model(
        csv_path='LLMAnnotation-gpt40-shashank.csv',
        model_name='meta-llama/Llama-3-8b',
        output_dir='wildercb/llama-rlhf-privacy-analyst',
        max_samples=5000
    )

if __name__ == "__main__":
    main()

