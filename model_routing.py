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
import aisuite as ai
import csv
from typing import List, Union, Optional

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