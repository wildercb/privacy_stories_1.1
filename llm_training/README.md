## A mini-repo for inferencing and finetuning llms from huggingface from local data

#### Data -
Contains scripts for example of how to load data into instruction tuning csvs from NLI data
Contains scripts to merge datasets for finetuning 

### Inference -
Scripts to load a model and inference from huggingface, working on adding ollama support - requires models to be stored in quantized format. 

for base inference

python inference/hf_inf_base.py --input_csv, --model_path (local or huggingface) , --n (number of prompts from input csv)

default saves to a csv which is named model_input_csv_n.csv (probably could be made better)

### Training - 
Scripts for finetuning and prompt tuning for choice of dataset and model 

For finetuning on huggingface data

python training/hf_finetune.py --base_model (model weights to start with (local or from huggingface)) --data_path (path to csv in instruction tuning format) --model_name (name of output model after finetuning) --checkpoint_dir (directory to save new weights) --batch_size , --num_epochs

### Models - 
Meant to store our different model training architectures and model argument parameters if needed in the future 


