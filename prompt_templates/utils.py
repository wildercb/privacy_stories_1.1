import json
from typing import Dict, List, Union
import tiktoken  
import sys
import os

notebook_dir = os.path.abspath(".")
project_root = os.path.abspath(os.path.join(notebook_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Calculate token count for the prompt
def count_tokens(prompt: str, model: str = "gpt-3.5-turbo") -> int:
    """Calculates the token count for a given prompt string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))
import json
from typing import Dict, Union, List

# Load the ontology from a JSON file
def load_privacy_ontology(ontology_path: str) -> Dict[str, Union[List[str], Dict]]:
    """Loads categories for actions, data types, and purposes from a JSON ontology file."""
    with open(ontology_path, 'r') as file:
        ontology = json.load(file)
    return ontology

def format_simple_category(category: Union[Dict, List]) -> str:
    """
    Formats a flat category (e.g., Actions or Purposes) including definitions if available.
    
    For each entry, if a "Description" is present, it will be appended to the category name.
    """
    if isinstance(category, list):
        return ", ".join(category)
    elif isinstance(category, dict):
        lines = []
        for key, details in category.items():
            desc = ""
            if isinstance(details, dict) and "Description" in details:
                if isinstance(details["Description"], list):
                    desc = " - " + " ".join(details["Description"])
                elif isinstance(details["Description"], str):
                    desc = " - " + details["Description"]
            lines.append(f"{key}{desc}")
        return "\n".join(lines)
    return ""

# Recursive function to display data types with subcategories including definitions
def format_data_types(data_types: Union[Dict, List], level: int = 0) -> str:
    """
    Formats data types and their subcategories recursively for prompt clarity,
    including definitions if provided.
    """
    formatted_text = ""
    indent = "  " * level  # Indentation for hierarchy

    if isinstance(data_types, list):
        formatted_text += indent + ", ".join(data_types) + "\n"
    elif isinstance(data_types, dict):
        for category, subcategories in data_types.items():
            # Check if the current entry has a definition
            description = ""
            if isinstance(subcategories, dict) and "Description" in subcategories:
                if isinstance(subcategories["Description"], list):
                    description = " - " + " ".join(subcategories["Description"])
                elif isinstance(subcategories["Description"], str):
                    description = " - " + subcategories["Description"]
            formatted_text += f"{indent}{category}{description}:\n"
            # Recurse into subcategories (ignore the "Description" key)
            if isinstance(subcategories, dict):
                for key, value in subcategories.items():
                    if key == "Description":
                        continue
                    formatted_text += format_data_types({key: value}, level + 1)
            elif isinstance(subcategories, list):
                formatted_text += indent + "  " + ", ".join(subcategories) + "\n"
    return formatted_text
