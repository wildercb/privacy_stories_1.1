import os
import sys
import json
import datetime
import argparse
from getpass import getpass

# Local imports
import utils.text_processing as tp
from model_routes import model_routing

def run_llm_annotations(
    prompts_data_json_file='data/annotations/prompts_data.json',
    default_models=None,
    default_num_runs=1
):
    """
    Creates/loads the prompts dictionary and runs multi-file annotations.
    
    Args:
        prompts_data_json_file (str): Path to the prompts data JSON file.
        default_models (list[str]): List of model names to run. If None, use a default list.
        default_num_runs (int): Number of runs per model. Defaults to 1.
    """
    # 1) Create/Load Human Annotation Data
    if not os.path.exists(prompts_data_json_file):
        prompt_templates_dict = tp.create_prompts_data_json(
            annotations_dir='data/annotations',
            ontology_path='privacy_ontology.json',
            output_json=prompts_data_json_file
        )
    else:
        print(f"Using existing prompts data JSON: {prompts_data_json_file}")
        with open(prompts_data_json_file, 'r', encoding='utf-8') as f:
            prompt_templates_dict = json.load(f)
    
    # 2) Prompt for API Keys
    os.environ['OPENAI_API_KEY'] = getpass('Enter your OPENAI API key: ')
    os.environ['GROQ_API_KEY']   = getpass('Enter your Groq API key: ')
    
    # 3) Models + runs
    if default_models is None or len(default_models) == 0:
        default_models = [
            'groq:qwen-2.5-32b',
            # 'openai:gpt-4o-latest',
        ]
    
    # Create a 6-character timestamp (HHMMSS)
    time_str = datetime.datetime.now().strftime('%H%M%S')
    
    # 4) Run annotations per model
    for model in default_models:
        # Make the model name file-safe
        model_filename = model.replace(':', '_').replace('.', '_')
        
        # Build a unique output CSV name
        output_csv = f"data/responses/LLMAnnotation_{model_filename}_{time_str}.csv"
        
        print(f"\nRunning annotation for model: {model}")
        print(f"Saving results to: {output_csv}")
        
        model_routing.run_multi_file_annotations(
            prompt_templates_dict,
            output_csv,
            models=[model],
            num_runs=default_num_runs
        )

def main():
    parser = argparse.ArgumentParser(description="Run LLM annotations with optional arguments.")
    parser.add_argument(
        '--models',
        nargs='+',
        help='List of model names (space-separated). E.g. --models groq:qwen-2.5-32b openai:gpt-4o-latest',
        default=None
    )
    parser.add_argument(
        '--num_runs',
        type=int,
        default=1,
        help='Number of runs per model.'
    )
    args = parser.parse_args()
    
    # Call our main function with the parsed arguments
    run_llm_annotations(
        default_models=args.models,
        default_num_runs=args.num_runs
    )

if __name__ == "__main__":
    main()
