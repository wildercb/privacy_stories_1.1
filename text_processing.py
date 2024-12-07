import os
import re
import os
import re
from typing import Dict, Union, List

# Updated function to clean the text

def clean_text(text: str) -> str:
    """Removes annotations (A:, DT:, P:, S:) and unwanted tags."""
    cleaned_text = re.sub(r'\(A:.*?\)|\(DT:.*?\)|\(P:.*?\)|\(S:.*?\)', '', text)
    return cleaned_text.strip()

# Updated function to process the new file format
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

# Recursive function to display data types with subcategories
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