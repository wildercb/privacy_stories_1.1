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
from typing import List, Union, Optional, Dict
import csv
import os
import json



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


def save_results_to_csv(models: List[str], 
                         prompt_templates_dict: Dict, 
                         model_responses: Dict, 
                         output_file: str):
    """
    Save the model outputs, prompts, and file annotations to a CSV file.
    """
    # Open the output CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # Prepare the CSV header
        fieldnames = ['File', 'Prompt', 'Model', 'Target File Path', 'Target Annotations', 'Model Response 1', 'Model Response 2']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Create a list of file keys to maintain order
        file_keys = list(prompt_templates_dict.keys())

        # Iterate through the prompt templates and models to pair responses
        for file_key in file_keys:
            template_info = prompt_templates_dict[file_key]
            
            for model in models:
                # Get responses for this model
                model_specific_responses = model_responses.get(model, {})
                
                # Get responses for this specific file/prompt
                prompt_responses = model_specific_responses.get(file_key, [])
                
                # Ensure we have at least two responses (or 'No Response')
                response1 = prompt_responses[0] if prompt_responses and len(prompt_responses) > 0 else 'No Response'
                response2 = prompt_responses[1] if prompt_responses and len(prompt_responses) > 1 else 'No Response'

                # Create a row for each model and file
                row = {
                    'File': file_key,
                    'Prompt': template_info['prompt_template'],
                    'Model': model,
                    'Target File Path': template_info['input_file_path'],
                    'Target Annotations': json.dumps(template_info['target_annotations']),
                    'Model Response 1': json.dumps(response1, ensure_ascii=False),
                    'Model Response 2': json.dumps(response2, ensure_ascii=False)
                }

                # Write the row to the CSV file
                writer.writerow(row)


def run_multi_file_annotations(prompt_templates_dict, output_csv, models=None):
    """
    Run annotation process for multiple files and save results to a CSV file.
    """

    try:
        # Create a list of file keys to maintain order
        file_keys = list(prompt_templates_dict.keys())

        # Prepare prompts that maintains the file key order
        test_prompts = [
            prompt_templates_dict[file_key]['prompt_template'] 
            for file_key in file_keys
        ]

        # Get model responses 
        raw_model_responses = run_multi_model_prompts(
            models=models, 
            prompts=test_prompts, 
            num_runs=2  # Number of runs per model
        )

        # Restructure responses to match file keys
        model_responses = {}
        for model in models:
            model_responses[model] = {
                file_key: model_runs 
                for file_key, model_runs in zip(file_keys, raw_model_responses.get(model, []))
            }

        # Save results to CSV 
        save_results_to_csv(
            models=models,
            prompt_templates_dict=prompt_templates_dict,
            model_responses=model_responses,
            output_file=output_csv
        )

        print(f"Annotation results saved to {output_csv}")
        print(f"Number of files processed: {len(prompt_templates_dict)}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

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