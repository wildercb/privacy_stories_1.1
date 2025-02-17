import os
import re
from typing import Dict, Union, List
import prompt_templates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

input_dir = 'input'

def clean_text(text: str) -> str:
    """Removes annotations (A:, DT:, P:, S:) and unwanted tags."""
    cleaned_text = re.sub(r'\(A:.*?\)|\(DT:.*?\)|\(P:.*?\)|\(S:.*?\)', '', text)
    return cleaned_text.strip()

def process_file(file_path: str) -> Dict:
    """Processes a single file to extract cleaned text, metadata, and stories."""
    with open(file_path, 'r') as file:
        text = file.read()

    # Clean the full text
    cleaned_text = clean_text(text)

    # Extract metadata (actions, data types, purposes, stories)
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

def format_data_types(data_types: Union[Dict, List], level: int = 0) -> str:
    """Formats data types and their subcategories recursively for prompt clarity."""
    formatted_text = ""
    indent = "  " * level  # Indentation for hierarchy

    if isinstance(data_types, list):
        formatted_text += indent + ", ".join(data_types) + "\n"
    elif isinstance(data_types, dict):
        for category, subcategories in data_types.items():
            formatted_text += f"{indent}{category}:\n"
            formatted_text += format_data_types(subcategories, level + 1)
    return formatted_text

def process_directory(directory_path):
    """Processes all text files in a directory and returns a list of dictionaries for each file."""
    results = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            file_data = process_file(file_path)
            results.append(file_data)
    return results

def process_input(path):
    """Determines if the path is a file or directory and processes accordingly."""
    if os.path.isdir(path):
        # Process all files in the directory
        return process_directory(path)
    elif os.path.isfile(path):
        # Process a single file
        return [process_file(path)]
    else:
        raise ValueError("Invalid path. Please provide a valid file or directory path.")


def find_matching_file(input_file_path, annotations_dir):
    """
    Find the matching annotation file for a given input file.
    
    Args:
        input_file_path (str): Path to the input file
        annotations_dir (str): Root directory of annotations
    
    Returns:
        str: Path to the matching annotation file, or None if not found
    """
    # Get the relative path from the input directory
    relative_path = os.path.relpath(input_file_path, input_dir)
    annotation_file_path = os.path.join(annotations_dir, relative_path)
    
    return annotation_file_path if os.path.exists(annotation_file_path) else None


def find_most_similar_file(input_file_path, annotations_dir, exclude_file):
    """
    Find the most similar annotation file to the input file, excluding a specified file.
    
    Args:
        input_file_path (str): Path to the input file
        annotations_dir (str): Directory containing the annotation files
        exclude_file (str): File to exclude from similarity search
    
    Returns:
        str: Path to the most similar annotation file
    """
    # Read the input file's content
    with open(input_file_path, 'r', encoding='utf-8') as f:
        input_text = f.read()
    
    # List to store (similarity_score, annotation_file_path)
    similarity_scores = []
    
    # Loop through the annotations directory to calculate similarity
    for root, _, files in os.walk(annotations_dir):
        for filename in files:
            if filename == exclude_file or filename.startswith('.'):
                continue
            
            annotation_file_path = os.path.join(root, filename)
            
            # Read the annotation file's content
            with open(annotation_file_path, 'r', encoding='utf-8') as f:
                annotation_text = f.read()
            
            # Compute similarity using TF-IDF and cosine similarity
            tfidf = TfidfVectorizer().fit_transform([input_text, annotation_text])
            similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            similarity_scores.append((similarity, annotation_file_path))
    
    # Sort by similarity score in descending order and return the most similar file
    similarity_scores.sort(reverse=True, key=lambda x: x[0])
    return similarity_scores[0][1] if similarity_scores else None


def create_prompt_templates_dict(input_dir='input', annotations_dir='annotations', ontology_path='privacy_ontology_simple.json'):
    """
    Create a dictionary of prompt templates matching input files with their corresponding annotation examples.
    
    Args:
        input_dir (str): Root directory containing input files to be annotated
        annotations_dir (str): Root directory containing annotated example files
        ontology_path (str): Path to the privacy ontology JSON file
    
    Returns:
        dict: Dictionary of prompt templates for each input file
    """
    # Load privacy ontology
    privacy_ontology = prompt_templates.load_privacy_ontology(ontology_path)
    
    # Dictionary to store prompt templates
    prompt_templates_dict = {}
    
    # Walk through all directories and files in the input directory
    for root, _, files in os.walk(input_dir):
        for filename in files:
            # Skip hidden files
            if filename.startswith('.'):
                continue
            
            # Full path to the input file
            input_file_path = os.path.join(root, filename)
            
            # Find corresponding annotation file (for exclusion from similarity search)
            annotation_file_path = find_matching_file(input_file_path, annotations_dir)
            
            if not annotation_file_path:
                print(f"No matching annotation found for {input_file_path}")
                continue
            
            try:
                # Read the input text to annotate
                with open(input_file_path, 'r', encoding='utf-8') as f:
                    new_text_to_annotate = f.read()
                
                # Find the most similar annotation file
                most_similar_annotation = find_most_similar_file(input_file_path, annotations_dir, os.path.basename(annotation_file_path))
                
                if not most_similar_annotation:
                    print(f"No similar annotation found for {input_file_path}")
                    continue
                
                # Process the example file from annotations
                example_file = process_input(most_similar_annotation)[0]
                
                # Create prompt template
                prompt_template = prompt_templates.create_annotation_prompt(
                    example_file, 
                    new_text_to_annotate, 
                    privacy_ontology
                )
                
                # Create a unique key based on relative path
                relative_key = os.path.relpath(input_file_path, input_dir)
                
                # Store in dictionary
                prompt_templates_dict[relative_key] = {
                    'input_file_path': input_file_path,
                    'annotation_file_path': most_similar_annotation,
                    'prompt_template': prompt_template,
                    'target_annotations': process_file(annotation_file_path),
                    'token_count': prompt_templates.count_tokens(prompt_template)
                }
            
            except Exception as e:
                print(f"Error processing {input_file_path}: {e}")
    
    return prompt_templates_dict


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