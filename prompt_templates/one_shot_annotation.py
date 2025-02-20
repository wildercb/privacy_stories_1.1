import utils
from typing import Dict, List, Union

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
    prompt += "<Actions>:\n" + utils.format_simple_category(ontology.get("Actions", {})) + "\n</Actions>\n"
    prompt += "<Data Types>:\n" + utils.format_data_types(ontology.get("Data Types", {})) + "\n</Data Types>\n"
    prompt += "<Purposes>:\n" + utils.format_simple_category(ontology.get("Purpose", {})) + "\n</Purposes>\n"
    
    prompt += "Refer to this example document and its corresponding annotations when making youre annotations:\n\n"
    prompt += f"{processed_file.get('full_cleaned_text', '')}\n\n"
    prompt += "Here are the behaviors and the privacy requirements in the form of privacy stories we found in this text:\n"
    prompt += "Privacy stories are built explicitly from our labelled privacy behaviors in the format: we (action) (data type) for (purpose).\n"
    
    if processed_file.get("metadata"):
        metadata = processed_file["metadata"]
        prompt += f"<Actions> {', '.join(metadata.get('actions', []))} </Actions> "
        prompt += f"<Data Types> {', '.join(metadata.get('data_types', []))} </Data Types> " 
        prompt += f"<Purposes> {', '.join(metadata.get('purposes', []))} </Purposes> "
        if metadata.get("stories"):
            stories_formatted = " ".join([f"{i+1}. {story}." for i, story in enumerate(metadata["stories"])])
            prompt += f"<Stories> {stories_formatted} </Stories>\n"
    
    prompt += "Here is youre document to annotate, when making these annotations refer only to the provided labels, only provide new labels where if they are absolutely necessary for generating the privacy requirement and explain how they would fit into the provided taxonomy of privacy behaviors. <Document> "
    prompt += f"{target_text} </Document>"
    prompt += (
        "Annotate the text with actions, data types, purposes, and stories as demonstrated, separating each category with their appropriate <Actions>, <Data Types>, <Purposes>, and <Stories> tags using only the categories from the list provided, please number your stories. For each annotation, provide your rationale and place <R> tag between your annotations and your rationale. Make sure to use this XML format properly in your response. Review the behaviors noted and stories generated provide only those which maximize accuracy and precision. Ensure to include your numbered stories in the format of we (action) (data type) for (purpose) inside of the <Stories> tags\n"
    )
    return prompt

