import re
from collections import Counter, defaultdict
from tqdm import tqdm
import torch
import pandas as pd

sanitizing_prompt="""
You are a text preprocessing assistant for a sentiment classification competition. Your job is to sanitize user-generated content while preserving its original meaning and sentiment.

Given an input text, follow these steps carefully:

1. If the text is in a language other than English, translate it into natural English.
2. Correct any spelling or grammatical errors.
3. Expand contractions (e.g., "don't" → "do not").
4. Replace any obfuscated or censored swearwords with their full uncensored versions, such as:
   - "A*#@le" → "asshole"
   - "m*ther f#cker" → "mother fucker"
   - (Handle other common masked profanity as well, using context to infer the intended word.)
5. Do not censor or euphemize offensive words—your goal is to retain the true intent of the sentence.
6. Do not classify or comment on the sentiment; only sanitize the text as described above.

Return only the cleaned and normalized sentence as output.

### Example

Input:  
A*#@le, you m*ther f#cker  

Output:  
Asshole, you mother fucker

---

Now process the following input:

Input: "<<INPUT>>"
"""

paraphrasing_prompt = """
You are helping with a sentiment classification task.

Given the following input sentence, generate exactly 2 paraphrased versions that express the same sentiment and meaning, but use different words or sentence structure.

Important rule:
- Always start each line with "<VARIATION> " followed by the paraphrased sentence.
- Return only the 2 lines.

Text: "<<INPUT>>"
"""


# """
# You are helping with a sentiment classification task.
#
# Given the following input sentence, generate 2 different paraphrased versions that express the same sentiment and meaning, but use different words or sentence structure.
#
# The goal is to improve diversity for classification without changing the tone or emotional content.
#
# Respond in this format:
#
# <VARIATION> <first paraphrased sentence>
# <VARIATION> <second paraphrased sentence>
#
# Text: "<<INPUT>>"
# """

insert_words_prompt = """
You are helping with a sentiment classification task.

Given the following sentence, generate 2 versions of the sentence by naturally inserting one or two words or short phrases. The added content should not change the sentiment or meaning, only provide variation.

Important rule:
- Always start each line with "<VARIATION> " followed by the paraphrased sentence.

Text: "<<INPUT>>"
"""

swap_words_prompt = """
You are helping with a sentiment classification task.

Given the following sentence, generate 2 versions by swapping words for suitable synonyms. The meaning and sentiment must remain exactly the same.

Important rule:
- Always start each line with "<VARIATION> " followed by the paraphrased sentence.

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

    outputs = pipeline(
        messages,
        max_new_tokens=max_new_tokens,
        pad_token_id=pipeline.tokenizer.eos_token_id,
    )

    return outputs[0]["generated_text"][-1]["content"]


def llm_augment(pipeline, df, prompt):
    """
    Generate variations for all texts in df using the prompt and pipeline.

    Args:
        pipeline: the LLM generation pipeline.
        df: DataFrame with at least a 'text' column.
        prompt: prompt string with <<INPUT>> placeholder.

    Returns:
        dict[int, list[str]]: key is variation index (0,1...), value is list of sentences.
    """
    variations_dict = defaultdict(list)

    for text in tqdm(df.text):
        prompt_filled = prompt.replace("<<INPUT>>", text)
        output_text = generate_variations(pipeline, prompt_filled, text)

        variations = extract_generated_variations(output_text)

        for i, variation in enumerate(variations):
            variations_dict[i].append(variation)

    return variations_dict


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


def save_variations_to_csv(variations_dict, df, prompt_name, base_path=""):
    """
    Save variations to CSV files, one file per variation index.

    Args:
        variations_dict: dict[int, list[str]]  # key=index of variation, value=list of sentences
        labels_dict: dict[int, list[int]]      # key=index of variation, value=list of labels for sentences
        base_path: str                         # directory + prefix for csv files

    CSV columns: id, sentence, label
    """
    for idx, sentences in variations_dict.items():
        df = pd.DataFrame(
            {
                "id": list(range(len(df))),
                "sentence": sentences,
                "labels": df["labels"],
            }
        )
        csv_file = f"test_{prompt_name}_variation_{idx+1}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved variation {idx+1} to {csv_file}")


def load_variations_csv(paths):
    """
    Load variations from multiple CSVs.

    Args:
        paths (list of str): List of CSV file paths, one per variation.

    Returns:
        List of lists: each sublist contains sentences for that variation index
    """
    all_variations = []
    for path in paths:
        df = pd.read_csv(path)
        sentences = df["sentence"].tolist()
        all_variations.append(sentences)
    return all_variations


def majority_vote(predictions):
    """
    Return the most common label from a list of predictions.
    """
    count = Counter(predictions)
    return count.most_common(1)[0][0]


@torch.inference_mode()
def predict_single_text(model, tokenizer, text, device, max_len=128):
    """
    Predict class index for a single text string.
    """
    model.eval()
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    _, pred = torch.max(logits, dim=1)
    return pred.item()


def predict_with_majority_voting(
    model, tokenizer, pipeline, original_text, prompt, device, variations_dict
):
    prompt_filled = prompt.replace("<<INPUT>>", original_text)
    output_text = generate_variations(pipeline, prompt_filled, original_text)
    variations = extract_generated_variations(output_text)

    all_texts = [original_text] + variations

    all_predictions = [
        predict_single_text(model, tokenizer, txt, device) for txt in all_texts
    ]

    final_prediction = majority_vote(all_predictions)

    return final_prediction, all_predictions
