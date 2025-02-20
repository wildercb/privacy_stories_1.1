## Privacy Stories Generator (Requirements Specification) 

This repository takes input software documentation to output a list of privacy behaviors and privacy requirements in the form of privacy stories. 


### Description 

Different parts of this repository will require diffent technical capabilities.
To prompt models for generation of privacy behaviors and stories one must first gather the appropriate API keys or access tokens for aisuite to use its model provides. 
For model finetuning will require gpu access, our fine-tuning code (utils/local_llm) (trainllama.py) may be run in a google colab notebook at 15-30 minutes per training step.

To run annotation prompt ```python run_prompts.py``` to use default parameters in file or 
```python run_llm_annotations.py --models groq:qwen-2.5-32b openai:gpt-4o-latest --num_runs 2``` to set manually models must in in [![AISuite](https://img.shields.io/badge/AI%20Suite-violet)](https://github.com/andrewyng/aisuite) format. API keys will be requested at runtime. 

# For Annotators: 

Annotating/Annotator_{model}.xlsx contains our annotation schema for answering questions related to privacy stories. 

Please refer to Annotating/README.md for instructions 

```python annotating/annotate.py``` can be used to open our annotation file allowing for preference choices. Assumes a csv that has 2 model responses per prompt. 

#### Data

data/annotations: 
This folder holds our annotations for text files of software documentation collected. 

data/responses:
This folder holds the llm responses.

data/preference_Data:
Here we store preference data for dpo training. 


#### Prompting 

notebooks/story_prompting.ipynb guides through the creation of the prompt template for generating privacy behaviors and stories, as well as sending these templates to our models. 
judge_prompting.ipynb guides through prompting for LM-LM evalutation. 


#### Evaluation 

utils/utils.ipynb walks through the evaluation of models for F1 score and ELO rating. 


#### Training
utils/local_llm/trainllama.py walks through our fine-tuning approach. utils.ipynb walks through the creation of training data from prefference data. llamainf.py will inference the model created from this training script with our task. 


### Breakdown of file structures:

##### model_routes:
This handles the api calls of prompting models. This file currently outputs responses in csv files and send api calls through aisuite and huggingface. 

##### prompt_templates:
These files hold functions to build various prompt templates used in the project. 

##### utils 
This folder holds local_llm inference and text processing required to create prompts. 







