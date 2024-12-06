import json
from typing import Dict, List, Union
import tiktoken  # Library for token counting with OpenAI models


# Calculate token count for the prompt
def count_tokens(prompt: str, model: str = "gpt-3.5-turbo") -> int:
    """Calculates the token count for a given prompt string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))

# Load the ontology from a JSON file
def load_privacy_ontology(ontology_path: str) -> Dict[str, Union[List[str], Dict]]:
    """Loads categories for actions, data types, and purposes from a JSON ontology file."""
    with open(ontology_path, 'r') as file:
        ontology = json.load(file)
    return ontology

# Recursive function to display data types with subcategories
def format_data_types(data_types: Union[Dict, List], level: int = 0) -> str:
    """Formats data types and their subcategories recursively for prompt clarity."""
    formatted_text = ""
    indent = "  " * level  # Indentation for hierarchy

    if isinstance(data_types, list):
        # Base case: data_types is a list of items
        formatted_text += indent + ", ".join(data_types) + "\n"
    elif isinstance(data_types, dict):
        # Recursive case: data_types is a dictionary with subcategories
        for category, subcategories in data_types.items():
            formatted_text += f"{indent}{category}:\n"
            formatted_text += format_data_types(subcategories, level + 1)
    return formatted_text

# Create the annotation prompt using the ontology
def create_annotation_prompt(example_file: Dict, target_text: str, ontology: Dict[str, Union[List[str], Dict]]) -> str:
    # Start building the prompt with an instructional message
    prompt = (
        "You are a privacy expert annotator who annotates text files with metadata about privacy behaviors in the form of "
        "actions, data types, and purposes. For each section in a file that contains behaviors related to each other, annotate the following:\n\n"
        "1. Actions: Actions that are performed or expected in this section.\n"
        "2. Data Types: Types of data referenced in this section. Data types may include specific subcategories.\n"
        "3. Purposes: Purposes or intentions related to these actions and data types.\n\n"
        "After providing your annotations, explain your rationale for these annotations. "
        "Place a <R> and </R> tags between your annotations and your rationale.\n\n"
    )
    # Add guidance from ontology
    prompt += "Use only the categories listed below when annotating the sections:\n\n"
    # Display actions
    prompt += "Actions:\n" + ", ".join(ontology.get("Actions", [])) + "\n\n"
    # Display data types with subcategories
    prompt += "Data Types:\n" + format_data_types(ontology.get("Data Types", {})) + "\n"
    # Display purposes
    prompt += "Purposes:\n" + ", ".join(ontology.get("Purpose", [])) + "\n\n"
    # Add an example from the provided file
    prompt += "Here is an example of annotated sections:\n\n"
    prompt += "Here is the section:\n"
    prompt += f"Full Cleaned Text:\n{example_file['full_cleaned_text']}\n\n"
    prompt += "And here are the sections you've annotated with their behaviors:\n\n"

    for section in example_file["sections"]:
        prompt += f"Section Text:\n{section['section_text_with_tags']}\n"
        prompt += f"Actions: {', '.join(section['metadata']['actions'] or [])}\n"
        prompt += f"Data Types: {', '.join(section['metadata']['data_types'] or [])}\n"
        prompt += f"Purposes: {', '.join(section['metadata']['purposes'] or [])}\n"
        if section.get('rationale') is not None: 
            prompt += "<R>\n"
            prompt += f"Rationale: {section.get('rationale')}\n\n"

    prompt += f"{target_text}\n\n"
    prompt += (
        "Annotate the sections of the above text with actions, data types, and purposes as demonstrated, "
        "using only the categories from the list provided. For each section, provide your annotations "
        "followed by your rationale, and place <R> and </R> tags between your annotations and your rationales.\n"
    )

    return prompt


# Create the annotation prompt using the ontology
def create_0_shot_annotation_prompt(example_file: Dict, target_text: str, ontology: Dict[str, Union[List[str], Dict]]) -> str:
    # Start building the prompt with an instructional message
    prompt = (
        "You are a privacy expert annotator who annotates text files with metadata about privacy behaviors in the form of "
        "actions, data types, and purposes. For each section in a file that contains behaviors related to each other, annotate the following:\n\n"
        "1. Actions: Actions that are performed or expected in this section.\n"
        "2. Data Types: Types of data referenced in this section. Data types may include specific subcategories.\n"
        "3. Purposes: Purposes or intentions related to these actions and data types.\n\n"
        "After providing your annotations, explain your rationale for these annotations. "
        "Place a <R> and </R> tags between your annotations and your rationale.\n\n"
    )
    # Add guidance from ontology
    prompt += "Use only the categories listed below when annotating the sections:\n\n"
    # Display actions
    prompt += "Actions:\n" + ", ".join(ontology.get("Actions", [])) + "\n\n"
    # Display data types with subcategories
    prompt += "Data Types:\n" + format_data_types(ontology.get("Data Types", {})) + "\n"
    # Display purposes
    prompt += "Purposes:\n" + ", ".join(ontology.get("Purpose", [])) + "\n\n"
            
    prompt += f"{target_text}\n\n"
    prompt += (
        "Annotate the sections of the above text with actions, data types, and purposes."
        "Using only the categories from the list provided. For each section, provide your annotations "
        "followed by your rationale, and place <R> and </R> tags between your annotations and your rationales.\n"
    )

    return prompt