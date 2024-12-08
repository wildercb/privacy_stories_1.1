''' For saving to csv 
def run_multi_model_prompts(
    models: List[str], 
    prompts: Union[str, List[str]], 
    num_runs: int = 3, 
    output_file: str = 'multi_model_comparison.csv',
    temperature: float = 0.7,
    max_tokens: int = 1500
):
    """
    Send prompts to multiple models and capture responses
    
    Args:
        models (List[str]): List of model identifiers
        prompts (Union[str, List[str]]): Single prompt or list of prompts to send
        num_runs (int): Number of times to run each model (default: 3)
        output_file (str): CSV file to save results (default: 'multi_model_comparison.csv')
        temperature (float): Sampling temperature for model responses (default: 0)
        max_tokens (int): Maximum number of tokens in response (default: 1500)
    """
    # Convert single prompt to list if needed
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # Initialize aisuite client
    client = ai.Client()
    
    # Prepare CSV file with headers
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # Header will be: Prompt, Model_Run1, Model_Run2, Model_Run3
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Prompt', 'Model'] + [f'Run{run}' for run in range(1, num_runs + 1)])
        
        # Iterate through prompts
        for prompt in prompts:
            # Iterate through models
            for model in models:
                # Prepare row with prompt and model
                row = [prompt, model]
                
                # Perform multiple runs for each model
                for _ in range(num_runs):
                    try:
                        # Prepare messages in the standard chat format
                        messages = [{"role": "system", "content": prompt}]
                        
                        # Create chat completion
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        
                        # Extract and add response content
                        response_text = response.choices[0].message.content.strip()
                        row.append(response_text)
                        
                        print(f"Model: {model} - Completed a run")
                    
                    except Exception as e:
                        print(f"Error with {model}: {e}")
                        row.append(f"ERROR: {str(e)}")
                
                # Write the row for this model and prompt
                csv_writer.writerow(row)
'''
from transformers import AutoModelForCausalLM, AutoTokenizer
import aisuite as ai
import csv
import torch
from typing import List, Union, Optional
import csv
import os


def run_multi_model_prompts(
    models: List[str], 
    prompts: Union[str, List[str]], 
    num_runs: int = 3, 
    temperature: float = 0.7,
    max_tokens: int = 1500
):
    """
    Send prompts to multiple models and capture responses

    Args:
        models (List[str]): List of model identifiers
        prompts (Union[str, List[str]]): Single prompt or list of prompts to send
        num_runs (int): Number of times to run each model (default: 3)
        temperature (float): Sampling temperature for model responses (default: 0)
        max_tokens (int): Maximum number of tokens in response (default: 1500)
    
    Returns:
        dict: A dictionary of model responses.
    """
    # Convert single prompt to list if needed
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # Initialize aisuite client
    client = ai.Client()

    # Store responses for all models
    model_responses = {model: [] for model in models}
    
    # Iterate through prompts
    for prompt in prompts:
        # Iterate through models
        for model in models:
            model_responses[model].append([])
            
            # Perform multiple runs for each model
            for _ in range(num_runs):
                try:
                    # Prepare messages in the standard chat format
                    messages = [{"role": "system", "content": prompt}]
                    
                    # Create chat completion
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    # Extract and add response content
                    response_text = response.choices[0].message.content.strip()
                    model_responses[model][-1].append(response_text)
                    
                    print(f"Model: {model} - Completed a run")
                
                except Exception as e:
                    print(f"Error with {model}: {e}")
                    model_responses[model][-1].append(f"ERROR: {str(e)}")
    
    return model_responses
        
def run_hf_prompts(
    models: List[str], 
    prompts: Union[str, List[str]], 
    num_runs: int = 3, 
    temperature: float = 0.7,
    max_tokens: int = 1500,
    output_dir: Optional[str] = None
):
    """
    Run prompts through multiple Hugging Face models and capture responses

    Args:
        models (List[str]): List of Hugging Face model identifiers
        prompts (Union[str, List[str]]): Single prompt or list of prompts to send
        num_runs (int): Number of times to run each model (default: 3)
        temperature (float): Sampling temperature for model responses (default: 0.7)
        max_tokens (int): Maximum number of tokens in response (default: 1500)
        output_dir (Optional[str]): Directory to save model responses (default: None)
    
    Returns:
        dict: A dictionary of model responses.
    """
    # Ensure output directory exists if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert single prompt to list if needed
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # Store responses for all models
    model_responses = {model: [] for model in models}
    
    # Iterate through prompts
    for prompt in prompts:
        # Iterate through models
        for model_name in models:
            model_responses[model_name].append([])
            
            try:
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
                
                # Move model to GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                
                # Perform multiple runs for each model
                for run in range(num_runs):
                    try:
                        # Prepare input
                        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                        
                        # Generate response
                        output = model.generate(
                            input_ids, 
                            max_length=input_ids.shape[1] + max_tokens,
                            num_return_sequences=1,
                            temperature=temperature,
                            do_sample=True
                        )
                        
                        # Decode and clean response
                        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
                        
                        # Remove the original prompt from the response
                        response_text = response_text[len(prompt):].strip()
                        
                        model_responses[model_name][-1].append(response_text)
                        
                        print(f"Model: {model_name} - Run {run + 1} completed")
                    
                    except Exception as run_error:
                        error_msg = f"ERROR in run {run + 1}: {str(run_error)}"
                        print(error_msg)
                        model_responses[model_name][-1].append(error_msg)
                
                # Optional: Save responses to CSV
                if output_dir:
                    save_responses_to_csv(model_responses, output_dir)
            
            except Exception as model_error:
                print(f"Error loading model {model_name}: {model_error}")
                model_responses[model_name][-1].append(f"ERROR LOADING MODEL: {str(model_error)}")
    
    return model_responses

def save_responses_to_csv(model_responses: dict, output_dir: str):
    """
    Save model responses to CSV files in the specified output directory

    Args:
        model_responses (dict): Dictionary of model responses
        output_dir (str): Directory to save CSV files
    """
    for model_name, prompt_responses in model_responses.items():
        # Create a safe filename by replacing invalid characters
        safe_model_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in model_name)
        csv_filename = os.path.join(output_dir, f"{safe_model_name}_responses.csv")
        
        # Write responses to CSV
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Prompt', 'Run', 'Response'])
            
            for prompt_idx, runs in enumerate(prompt_responses):
                for run_idx, response in enumerate(runs):
                    csv_writer.writerow([f"Prompt {prompt_idx + 1}", f"Run {run_idx + 1}", response])
        
        print(f"Responses for {model_name} saved to {csv_filename}")

# Example usage
if __name__ == "__main__":
    # Example list of models (replace with actual Hugging Face model names)
    model_list = [
        "gpt2",  # Small GPT-2 model
        "distilgpt2",  # Distilled version of GPT-2
        "EleutherAI/gpt-neo-125M"  # GPT-Neo 125M model
    ]
    
    prompts = [
        "Tell me a short story about a brave adventurer.",
        "Explain the basics of quantum computing in simple terms."
    ]
    
    # Run models with responses saved to 'model_outputs' directory
    responses = run_multi_model_prompts(
        models=model_list, 
        prompts=prompts, 
        num_runs=2, 
        output_dir="model_outputs"
    )
'''
import openai
import json
from typing import Dict, List, Union
import tiktoken  # Library for token counting with OpenAI models
from secrets_file import openai_api_key
import csv
import torch
from prompt_templates import load_privacy_ontology, count_tokens, create_annotation_prompt
from text_processing import process_input

def send_prompt(prompt: str, model_name: str, use_openai: bool = True, **kwargs):
    if use_openai:
        # Set OpenAI API key
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "system", "content": prompt}],
            temperature=kwargs.get('temperature', 0),
            max_tokens=kwargs.get('max_tokens', 1500)
        )
        output = response.choices[0].message.content.strip()
    else:
        # Use Hugging Face Mamba 2 model
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Load tokenizer with specific settings
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision='refs/pr/9', from_slow=True, legacy=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, revision='refs/pr/9')
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Encode prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate output
        input_length = inputs['input_ids'].shape[1]
        max_length = input_length + kwargs.get('max_tokens', 150)
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=kwargs.get('temperature', 0.7),
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract generated text after the prompt
        output = full_output[len(prompt):].strip()
    return output

# Function to create a few-shot prompt with multiple example files and annotate a new target file
def annotate_with_few_shot_prompt(example_directory: str, target_file_path: str, ontology_path: str, model_name: str = "gpt-4", use_openai: bool = True):
    # Load the privacy ontology
    ontology = load_privacy_ontology(ontology_path)
    
    # Load all example files in the directory
    example_files = process_input(example_directory)
    
    # Load the content of the target text file
    with open(target_file_path, 'r') as target_file:
        target_text = target_file.read()
    
    # Generate the few-shot prompt using `create_annotation_prompt`
    prompt = ""
    for example_file in example_files:
        prompt += create_annotation_prompt(example_file, target_text, ontology) + "\n\n"
    
    # Print prompt and token count for review
    print(prompt)
    if use_openai:
        token_count = count_tokens(prompt)
    else:
        # Load tokenizer for Hugging Face model
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision='refs/pr/9', from_slow=True, legacy=False)
        token_count = count_tokens(prompt, tokenizer=tokenizer)
    print(f"\nToken Count: {token_count} tokens")
    
    # Send the prompt to the chosen LLM
    annotated_data = send_prompt(
        prompt=prompt,
        model_name=model_name,
        use_openai=use_openai,
        temperature=0,
        max_tokens=1500
    )
    
    print("\nAnnotations from LLM:\n", annotated_data)
    
    # Save the prompt and response to CSV
    with open('llm_output.csv', mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([prompt, annotated_data])
'''