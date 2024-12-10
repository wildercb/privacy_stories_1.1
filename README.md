## Privacy Stories Generator (Requirements Specification) 

This repository takes input software documentation to output a list of privacy behaviors and privacy requirements in the form of privacy stories. 


### Description 

Different parts of this repository will require diffent technical capabilities.
To prompt models for generation of privacy behaviors and stories one must first gather the appropriate API keys or access tokens for aisuite to use its model provides. 
For model finetuning will require gpu access, our fine-tuning code (trainllama.py) may be run in a google colab notebook at 15-30 minutes per training step.

#### Data
The directory input is a copy of the directory annotations except for without the annotations (these directories are our documentation data)

#### Prompting 

story_prompting.ipynb guides through the creation of the prompt template for generating privacy behaviors and stories, as well as sending these templates to our models. 
judge_prompting guides through prompting for LM-LM evalutation. 


#### Evaluation 

utils.ipynb walks through the evaluation of models for F1 score and ELO rating. 

```python annotate.py``` can be used to open our annotation file allowing for preference choices.

#### Training
trainllama.py walks through our fine-tuning approach. utils.ipynb walks through the creation of training data from prefference data. 


### Requirements 

#### Functional Requirements:

##### I/O
The system may process any text file as software documentation 
The system shall output the generated privacy behaviors and requirements in a parsable csv 

##### Prompting
The system allows for management, refinement and creation of prompt templates (prompt_templates.py)

##### Eval
The system provides F1 evaluations across input documentation datasets

Non-Functional Requirements 

##### Training
The system shall allow for fine-tuning of small (<7b param) LMs efficiently such that they can be currently used in Google Colab or Kaggle Notebooks. 

##### Usability

The system shall provide user-friendly notebooks with descriptions that allow someone with no coding experience to use them.







