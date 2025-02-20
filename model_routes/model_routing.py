from transformers import AutoModelForCausalLM, AutoTokenizer
import aisuite as ai
import csv
import torch
from typing import List, Dict, Union, Optional
import os
import json

def run_multi_model_prompts(
    models: List[str], 
    prompt: str,   # Single prompt string
    num_runs: int = 3, 
    temperature: float = 0.7,
    max_tokens: int = 1500
) -> Dict[str, List[str]]:
    """
    Sends a single prompt to multiple models and captures their responses.
    """
    client = ai.Client()
    responses = {model: [] for model in models}
    
    for model in models:
        for _ in range(num_runs):
            try:
                messages = [{"role": "user", "content": prompt}]
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                response_text = response.choices[0].message.content.strip()
                responses[model].append(response_text)
                print(f"Model: {model} - Completed a run for prompt starting with: {prompt[:30]}...")
            except Exception as e:
                print(f"Error with {model}: {e}")
                responses[model].append(f"ERROR: {str(e)}")
    
    return responses

def run_multi_file_annotations(
    prompt_templates_dict: Dict, 
    output_csv: str, 
    models: List[str], 
    num_runs: int = 1
):
    """
    Processes each file’s prompt one by one and writes the results to CSV.
    """
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            base_fieldnames = ['File', 'Prompt', 'Model', 'Target File Path', 'Target Annotations']
            response_fieldnames = [f"Model Response {i+1}" for i in range(num_runs)]
            fieldnames = base_fieldnames + response_fieldnames
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            file_keys = sorted(prompt_templates_dict.keys())
            for file_key in file_keys:
                prompt_info = prompt_templates_dict[file_key]
                prompt = prompt_info['prompt_template']
                
                model_responses = run_multi_model_prompts(
                    models=models,
                    prompt=prompt,
                    num_runs=num_runs
                )
                
                for model in models:
                    responses = model_responses.get(model, [])
                    row = {
                        'File': file_key,
                        'Prompt': prompt,
                        'Model': model,
                        'Target File Path': prompt_info.get('annotation_file_path', ''),
                        'Target Annotations': json.dumps(prompt_info.get('target_annotations', {}))
                    }
                    
                    for i in range(num_runs):
                        key = f"Model Response {i+1}"
                        row[key] = json.dumps(responses[i] if i < len(responses) else 'No Response', ensure_ascii=False)
                    
                    writer.writerow(row)
                    csvfile.flush()
                
                print(f"Processed file: {file_key}")
            
            print(f"Annotation results saved to {output_csv}")
            print(f"Number of files processed: {len(file_keys)}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# Function to generate prompts from model_routing CSV file and send to a single model
def llm_llm_eval(csv_file_path: str, model: str, num_runs: int = 1, temperature: float = 0.7, max_tokens: int = 50):
    """Reads a CSV file, generates judge prompts, sends them to a single model, and saves responses to a new CSV file."""
    
    # Read CSV and generate prompts
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        input_rows = list(reader)
        fieldnames = reader.fieldnames + ['AI Preference']  # Add new column

    prompts = []
    for row in input_rows:
        original_prompt = row.get('Prompt', '')
        response_1 = row.get('Model Response 1', '')
        response_2 = row.get('Model Response 2', '')

        prompt = prompt_templates.create_judge_prompt(original_prompt, response_1, response_2)
        prompts.append((row, prompt))  # Keep the original row paired with the prompt

    # Initialize aisuite client
    client = ai.Client()

    # Prepare to store responses
    updated_rows = []

    # Iterate over the prompts and get responses from the model
    for row, prompt in prompts:
        ai_preference = None

        # Run the prompt multiple times
        for _ in range(num_runs):
            try:
                # Prepare the message for the model
                messages = [{"role": "system", "content": prompt}]
                
                # Send the prompt to the model via aisuite
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # Extract the response and use it as the AI preference
                ai_preference = response.choices[0].message.content.strip()
                print(f"AI Preference: {ai_preference}")
                break  # Exit loop on successful response
            except Exception as e:
                print(f"Error with model {model}: {e}")
                ai_preference = f"ERROR: {str(e)}"

        # Update the row with the AI Preference
        row['AI Preference'] = ai_preference
        updated_rows.append(row)

    # Save the updated rows to a new CSV file
    output_file_path = os.path.splitext(csv_file_path)[0] + f'_with_ai_preference_{model}'
    with open(output_file_path, mode='w', encoding='utf-8', newline='') as outputfile:
        writer = csv.DictWriter(outputfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Processed file saved to {output_file_path}")

def run_hf_prompts(
    models: List[str], 
    prompts: Union[str, List[str]], 
    num_runs: int = 3, 
    temperature: float = 0.7,
    max_tokens: int = 1500,
    output_dir: Optional[str] = None
):
    """
    Run prompts through Hugging Face models locally 

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

