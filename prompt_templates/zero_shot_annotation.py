import utils 
from typing import Dict, List, Union

def create_0_shot_annotation_prompt(
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
    # Add guidance from ontology
    prompt += "Use only the categories listed below when annotating the sections:\n\n"
    prompt += "Actions:\n" + utils.format_simple_category(ontology.get("Actions", {})) + "\n\n"
    prompt += "Data Types:\n" + utils.format_data_types(ontology.get("Data Types", {})) + "\n"
    prompt += "Purposes:\n" + utils.format_simple_category(ontology.get("Purpose", {})) + "\n\n"
            
    prompt += f"{target_text}\n\n"
    prompt += (
        "Annotate the sections of the above text with actions, data types, and purposes using only the categories from the list provided. "
        "For each section, provide your annotations followed by your rationale, and place <R> and </R> tags between your annotations and your rationale.\n"
    )
    return prompt



