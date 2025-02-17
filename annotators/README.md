### To annotate privacy stories: 

## Rules:

Each story for each document has 4 questions, additionally each document has 1 final question in regards to all of these stories.

Each stories questions are grouped in 4 column, For the first two select an item in the drop down menu. For the Second input text if applicable. 

## Guideline: 

When annotating, either review the target annotations column, note how the target text, as well as our annotations for that text are contained as json in this column. Alternatively use this repository to find the text file found in the target path column. 

# Story Questions 

Question 1: Please review our annotations, and if there are additional behaviors present in the LLMs response review the file text and verify whether or not these are accurate. If any are not accurate for that story the answer is simply No, else it should be Yes. 

Note: Sometimes the LLM may output labels which are not contained in the taxonomy this may be a hallucination however, in this case it is up to your judgement to determine whether this label would make sense to in the taxonomy as well as be accurate within the story. 
Additionally sometimes the LLMs may expand, in a scenario like this "5. We use health data (CCR/CCD) and user id for security." Where they name the label with a slight bit more detail is correct and what we hope to see. Cases like this "We use phone numbers for internal reference and email addresses for sending password reset links and other communications." Are outside of the structure of privacy stories and may be accurate but should be rewritten.

Question 2: If yes to the previous question (otherwise leave blank), Select yes if any behaviors which are more precise (i.e., a lower level in the privacy_ontology_simple.json) should have been used in this story instead. For example if the story contains the purpose of 'Security' where it could have more precisely included the behavior of 'Fraud Prevention'select yes, else select no. 

Question 3: Here if yes to the previous question (otherwise leave blank), write in the more precise beahviors which should have been present, if this is multiple please seperate with ,. 

Question 4 (Optional): Here, rewrite the story to be as accurate and precise as possible.  

# General Question (Found in final row of annotator files).

Please write any stories which are missing, review our annotations and/or the file itself to determine if any stories were missing from LLM response and write these in. 



## Why are you doing this? 

A. Evaluate the accuracy of base LLMs at generating privacy requirements given code documentation / follow an annotation schema. 

B. Creating training data to finetune LLMs further at this task. 

