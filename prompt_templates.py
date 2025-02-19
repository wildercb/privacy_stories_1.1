import json
from typing import Dict, List, Union
import tiktoken  


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

# Updated function to create the annotation prompt
def create_annotation_prompt(
    processed_file: Dict, target_text: str, ontology: Dict[str, Union[List[str], Dict]]
) -> str:
    prompt = (
        "Your mission is to annotate software documentation with privacy behaviors in the form of actions, data types and purposes to build privacy requirements in the form of privacy stories. "
        "For the given text, provide the following: "
        "1. <Actions>: Actions performed or expected in the text. "
        "2. <Data Types>: Types of data referenced in the text. Data types may include specific subcategories. "
        "3. <Purposes>: Intentions or purposes related to the actions and data types. "
        "4. <Stories>: Concise stories that describe how actions, data types, and purposes interact in context. "
        "After providing your annotations, explain your rationale for these annotations. "
        "Place <R> tag between your annotations and your rationale.\n"
    )
    # Add ontology guidance
    prompt += "Use only the categories listed below when annotating:\n"
    prompt += "<Actions>:\n" + format_simple_category(ontology.get("Actions", {})) + "\n</Actions>\n"
    prompt += "<Data Types>:\n" + format_data_types(ontology.get("Data Types", {})) + "\n</Data Types>\n"
    prompt += "<Purposes>:\n" + format_simple_category(ontology.get("Purpose", {})) + "\n</Purposes>\n"
    
    prompt += "Here is the text:\n\n"
    prompt += f"{processed_file.get('full_cleaned_text', '')}\n\n"
    prompt += "Here are the behaviors and the privacy requirements in the form of privacy stories we build from them:\n"
    prompt += "Privacy stories are built explicitly from our labelled privacy behaviors in the format: we (action) (data type) for (purpose).\n"
    
    if processed_file.get("metadata"):
        metadata = processed_file["metadata"]
        prompt += f"<Actions> {', '.join(metadata.get('actions', []))} </Actions> "
        prompt += f"<Data Types> {', '.join(metadata.get('data_types', []))} </Data Types> " 
        prompt += f"<Purposes> {', '.join(metadata.get('purposes', []))} </Purposes> "
        if metadata.get("stories"):
            stories_formatted = " ".join([f"{i+1}. {story}." for i, story in enumerate(metadata["stories"])])
            prompt += f"<Stories> {stories_formatted} </Stories>"
    prompt += f"{target_text}\n\n"
    prompt += (
        "Annotate the text with actions, data types, purposes, and stories as demonstrated, separating each category with their "
        "appropriate <Actions>, <Data Types>, <Purposes>, and <Stories> tags using only the categories from the list provided. "
        "For each annotation, provide your rationale and place <R> tag between your annotations and your rationale. "
        "Make sure to use this XML format properly in your response.\n"
    )
    return prompt

def create_0_shot_annotation_prompt(
    example_file: Dict, target_text: str, ontology: Dict[str, Union[List[str], Dict]]
) -> str:
    # Start building the prompt with an instructional message
    prompt = (
        "You are a privacy expert annotator who annotates text files with metadata about privacy behaviors in the form of "
        "actions, data types, and purposes. For each section in a file that contains behaviors related to each other, annotate the following:\n\n"
        "1. Actions: Actions that are performed or expected in this section.\n"
        "2. Data Types: Types of data referenced in this section. Data types may include specific subcategories.\n"
        "3. Purposes: Purposes or intentions related to these actions and data types.\n\n"
        "After providing your annotations, explain your rationale for these annotations. "
        "Place <R> and </R> tags between your annotations and your rationale.\n\n"
    )
    # Add guidance from ontology
    prompt += "Use only the categories listed below when annotating the sections:\n\n"
    prompt += "Actions:\n" + format_simple_category(ontology.get("Actions", {})) + "\n\n"
    prompt += "Data Types:\n" + format_data_types(ontology.get("Data Types", {})) + "\n"
    prompt += "Purposes:\n" + format_simple_category(ontology.get("Purpose", {})) + "\n\n"
            
    prompt += f"{target_text}\n\n"
    prompt += (
        "Annotate the sections of the above text with actions, data types, and purposes using only the categories from the list provided. "
        "For each section, provide your annotations followed by your rationale, and place <R> and </R> tags between your annotations and your rationale.\n"
    )
    return prompt


# Create the judge prompt template
def create_judge_prompt(original_prompt: str, response_1: str, response_2: str) -> str:
    prompt = (
        "You are an impartial judge evaluating responses to a given prompt. Your task is to assess the quality of two responses and select the better one.\n\n"
        "Consider the following criteria when making your decision:\n"
        "1. Completeness: Does the response fully address the original prompt? Do they hallucinate behaviors if so mark down, the biggest flaw of models at this task is over completeness.\n"
        "2. Clarity: Is the response easy to understand and well-organized in a way that clearly displays privacy behaviors and stories?\n"
        "3. Accuracy: Is the information provided correct and relevant in accordance to the original file.\n"
        "Explicitly output \"1\" if Response 1 is better, and \"2\" if Response 2 is better."
    )
    prompt += "\n\nHere is the original prompt:\n\n"
    prompt += f"{original_prompt}\n\n"

    prompt += "Here are the responses:\n\n"
    prompt += "Response 1:\n\n"
    prompt += f"{response_1}\n\n"

    prompt += "Response 2:\n\n"
    prompt += f"{response_2}\n\n"

    prompt += (
        "Considering the criteria listed above, which response is better? Provide your answer explicitly as \"1\" or \"2\"."
    )

    return prompt

