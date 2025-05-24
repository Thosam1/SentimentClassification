import re

paraphrasing_prompt = """
You are helping with a sentiment classification task.

Given the following input sentence, generate 2 different paraphrased versions that express the same sentiment and meaning, but use different words or sentence structure.

The goal is to improve diversity for classification without changing the tone or emotional content.

Respond in this format:

<VARIATION> <first paraphrased sentence>
<VARIATION> <second paraphrased sentence>

Text: "<<INPUT>>"
"""

insert_words_prompt = """
You are helping with a sentiment classification task.

Given the following sentence, generate 2 versions of the sentence by naturally inserting one or two words or short phrases. The added content should not change the sentiment or meaning, only provide variation.

Keep the tone and emotional content the same.

Respond in this format:

<VARIATION> <first modified sentence>
<VARIATION> <second modified sentence>

Text: "<<INPUT>>"
"""

swap_words_prompt = """
You are helping with a sentiment classification task.

Given the following sentence, generate 2 versions by swapping words for suitable synonyms. The meaning and sentiment must remain exactly the same.

Only make small changes that improve diversity.

Respond in this format:

<VARIATION> <first modified sentence>
<VARIATION> <second modified sentence>

Text: "<<INPUT>>"
"""


def generate_variations(
    pipeline, prompt_instruction, text, system_instruction=None, max_new_tokens=256
):
    """
    Generate output variations for sentiment data augmentation.

    Args:
        pipeline (Callable): A chat completion or generation pipeline.
        prompt_instruction (str): Prompt template containing 'seed text placeholder'.
        text (str): The input text to substitute into the prompt.
        system_instruction (str, optional): Optional system message to guide the assistant.
        max_new_tokens (int): Number of new tokens to generate.

    Returns:
        str: The LLM-generated response text.
    """
    # Replace the placeholder with the actual text
    user_prompt = prompt_instruction.replace("<<INPUT>>", text)

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    messages.append({"role": "user", "content": user_prompt})

    outputs = pipeline(messages, max_new_tokens=max_new_tokens, pad_token_id=pipeline.tokenizer.eos_token_id)

    return outputs[0]["generated_text"][-1]["content"]


def extract_generated_variations(response_text):
    """
    Extracts sentence variations tagged with <VARIATION> from a model-generated response.

    Args:
        response_text (str): The text output from the model.

    Returns:
        List[str]: A list of extracted sentences.
    """
    # Pattern to match lines starting with <VARIATION> followed by a space and the sentence
    pattern = r"<VARIATION>\s*(.*)"

    matches = re.findall(pattern, response_text)

    # fallback if no matches found, keep the original logic (optional)
    if not matches:
        # fallback to previous numbered extraction
        pattern_num = r'\d+\.\s*["“](.*?)["”](?=\s*\d+\.|$)'
        matches = re.findall(pattern_num, response_text, re.DOTALL)

        if not matches:
            lines = response_text.strip().split("\n")
            matches = [
                line.split(". ", 1)[-1].strip().strip('"“”')
                for line in lines
                if ". " in line
            ]

    return matches
