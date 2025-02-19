import os
import re
import json
from typing import Dict, List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import prompt_templates  # Must provide load_privacy_ontology, create_annotation_prompt, count_tokens

def clean_text(text: str) -> str:
    """Removes annotations (A:, DT:, P:, S:) and unwanted tags."""
    cleaned_text = re.sub(r'\(A:.*?\)|\(DT:.*?\)|\(P:.*?\)|\(S:.*?\)', '', text)
    return cleaned_text.strip()

def process_file(file_path: str) -> Dict:
    """Processes a single file to extract cleaned text, metadata, and stories."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    cleaned_text = clean_text(text)
    actions = re.findall(r'\(A:\s*(.*?)\)', text)
    data_types = re.findall(r'\(DT:\s*(.*?)\)', text)
    purposes = re.findall(r'\(P:\s*(.*?)\)', text)
    stories = re.findall(r'\(S:\s*(.*?)\)', text)
    return {
        "file_name": os.path.basename(file_path),
        "full_cleaned_text": cleaned_text,
        "metadata": {
            "actions": actions if actions else None,
            "data_types": data_types if data_types else None,
            "purposes": purposes if purposes else None,
            "stories": stories if stories else None,
        }
    }

def build_annotations_data(annotations_dir: str) -> Dict:
    """
    Processes all annotation files in the directory and returns a dictionary
    mapping each file’s relative path (from annotations_dir) to its processed data.
    """
    annotations_data = {}
    for root, _, files in os.walk(annotations_dir):
        for filename in files:
            if filename.startswith('.'):
                continue
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, annotations_dir)
            try:
                annotations_data[rel_path] = process_file(file_path)
            except Exception as e:
                print(f"Error processing annotation file {file_path}: {e}")
    return annotations_data

def create_prompts_data_json(annotations_dir: str = 'data/annotations', 
                               ontology_path: str = 'privacy_ontology_simple.json', 
                               output_json: str = 'data/annotations/prompts_data.json') -> Dict:
    """
    Processes the annotations directory and creates a single JSON file containing, for each
    annotation file:
      - annotation_file_path
      - prompt_template (generated from the annotation’s own processed data)
      - target_annotations (the processed annotation data)
      - token_count, etc.
    """
    # Load privacy ontology from prompt_templates module.
    privacy_ontology = prompt_templates.load_privacy_ontology(ontology_path)
    
    # Build annotations data.
    annotations_data = build_annotations_data(annotations_dir)
    
    prompts_data = {}
    for rel_path, annotation_data in annotations_data.items():
        full_annotation_path = os.path.join(annotations_dir, rel_path)
        try:
            # Use the file's own processed text as both the example and the text to annotate.
            new_text = annotation_data.get('full_cleaned_text', '')
            prompt_template = prompt_templates.create_annotation_prompt(annotation_data, new_text, privacy_ontology)
            token_count = prompt_templates.count_tokens(prompt_template)
            
            prompts_data[rel_path] = {
                'annotation_file_path': full_annotation_path,
                'prompt_template': prompt_template,
                'target_annotations': annotation_data,
                'token_count': token_count
            }
        except Exception as e:
            print(f"Error processing {full_annotation_path}: {e}")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(prompts_data, f, indent=2)
    print(f"Prompts data JSON saved to {output_json}")
    return prompts_data


'''To work with sectioned annotations 
def clean_full_text(text):
    """Removes annotations (A:, DT:, P:), section tags {#s ... \}, and <PI> tags from the text."""
    cleaned_text = re.sub(r'\(A:.*?\)|\(DT:.*?\)|\(P:.*?\)', '', text)
    cleaned_text = re.sub(r'{#s|\}', '', cleaned_text)
    cleaned_text = re.sub(r'<PI>', '', cleaned_text)
    return cleaned_text.strip()

def process_file(file_path):
    """Processes a single file to extract cleaned full text and sections with metadata."""
    with open(file_path, 'r') as file:
        text = file.read()

    # Full cleaned text without annotations or section tags
    cleaned_full_text = clean_full_text(text)
    
    # Prepare dictionary structure
    result = {
        "file_name": os.path.basename(file_path),
        "full_cleaned_text": cleaned_full_text,
        "sections": []
    }

    # Extract sections defined by {#s ... \}
    section_pattern = re.compile(r'{#s(.*?)\\}', re.DOTALL)
    sections = section_pattern.findall(text)

    for section in sections:
        # Clean section text by removing annotations
        cleaned_section = re.sub(r'\(A:.*?\)|\(DT:.*?\)|\(P:.*?\)', '', section).strip()

        # Extract metadata (actions, data types, purposes)
        actions = re.findall(r'\(A:\s*(.*?)\)', section)
        data_types = re.findall(r'\(DT:\s*(.*?)\)', section)
        purposes = re.findall(r'\(P:\s*(.*?)\)', section)
        
        # Add section data to dictionary
        result["sections"].append({
            "section_text_with_tags": cleaned_section,
            "cleaned_section_text": cleaned_section,
            "metadata": {
                "actions": actions if actions else None,
                "data_types": data_types if data_types else None,
                "purposes": purposes if purposes else None
            }
        })
        
    return result
'''