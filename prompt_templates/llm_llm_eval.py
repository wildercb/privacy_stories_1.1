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

