from datasets import load_dataset
import torch
import re

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SYSTEM_PROMPT_2 = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: prompt. Assistant:
"""


def extract_answer_from_model_output(text):
    """
    Extracts the answer from model output using three main patterns:
    1. XML tags <answer>...</answer>
    2. LaTeX boxed format
    3. Conclusion sentences with markers like "therefore", "thus", etc.
    
    Args:
        text (str): The model-generated text containing the answer.
        
    Returns:
        str or None: The extracted answer, or None if no valid answer is found.
    """
    # Pattern 1: Extract from <answer> tags
    answer = extract_from_answer_tags(text)
    if answer and answer[-1] != '\n...\n':
        # print("get <answer> tag answer", answer)
        # print("but use the last answer")
        return answer[-1]  # For evaluation, use the last answer
    
    # Pattern 2: Extract from LaTeX boxed format (get list of answers)
    boxed_answers = extract_from_latex_boxed(text)
    if boxed_answers:
        # print("get boxed answer", boxed_answers)
        # print("but use the first answer")
        return boxed_answers  # For evaluation, use the first answer
    
    # Pattern 3: Extract from conclusion sentences
    answer = extract_from_conclusion_sentences(text)
    if answer:
        # print("get conclusion answer", answer)
        # print("but use the first answer")
        return answer  # For evaluation, use the last answer
    
    # If all methods fail, return None
    return None

def extract_from_answer_tags(text):
    """
    Extract contains between XML tags: <answer>...</answer> from the text.
    Returns a list of answers found in the tags, or an empty list if none found.
    """
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    answer_matches = re.findall(answer_tag_pattern, text, re.DOTALL)
    return answer_matches



    # answer_tag_pattern = r'<answer>(.*?)</answer>'
    # answer_matches = re.findall(answer_tag_pattern, text, re.DOTALL)
    
    # if not answer_matches:
    #     return []
    
    # Process each answer match
    # answers = []
    # for match in answer_matches:
    #     content = match.strip()
    #     if content and content != "...":
    #         # If the content contains numbers, extract the first one
    #         numbers = re.findall(r'-?\d+\.?\d*', content)
    #         if numbers:
    #             answers.append(numbers[0])
    #         else:
    #             # If no numbers found, add the entiretent con
                # answers.append(content)
    
def extract_from_latex_boxed(text):
    """
    Extract all answers from LaTeX boxed format: \[\boxed{...}\] or \boxed{...}
    
    Args:
        text (str): The model-generated text containing LaTeX boxed answers.
        
    Returns:
        list: A list of all boxed answers found, or empty list if none found.
    """
    # Check for LaTeX boxed format with number content
    boxed_pattern = r'\\boxed\s*\{\s*(.*?)\s*\}'
    boxed_matches = re.findall(boxed_pattern, text)
    
    # Return all boxed answers as a list
    if boxed_matches:
        return [match.strip() for match in boxed_matches]
    
    return []

def extract_from_conclusion_sentences(text):
    """
    Extract answers from sentences with conclusion markers.
    High priority markers take precedence over regular conclusion markers.
    """
    # Define high priority markers
    high_priority_markers = [
        'the final answer is', 'final answer'
    ]
    
    # Define regular conclusion markers
    conclusion_markers = [
        'therefore', 'thus', 'hence', 'so', 'in conclusion', 'finally',
        'consequently', 'as a result', 'this means', 'this gives'
    ]
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Store all numbers found after markers
    high_priority_numbers = []
    conclusion_numbers = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # First check for high priority markers
        high_priority_match = False
        for marker in high_priority_markers:
            # Use word boundary \b to match whole words only
            pattern = r'\b' + re.escape(marker) + r'\b'
            if re.search(pattern, sentence_lower):
                high_priority_match = True
                
                # Find the position of the marker with word boundaries
                marker_match = re.search(pattern, sentence_lower)
                marker_pos = marker_match.start()
                
                # Extract the part of the sentence after the marker
                text_after_marker = sentence[marker_pos + len(marker):]
                
                # Find the first number in this part
                number_match = re.search(r'-?\d+\.?\d*', text_after_marker)
                if number_match:
                    high_priority_numbers.append(number_match.group(0))
        
        # Only check regular conclusion markers if no high priority markers were found
        if not high_priority_match:
            for marker in conclusion_markers:
                # Use word boundary \b to match whole words only
                pattern = r'\b' + re.escape(marker) + r'\b'
                if re.search(pattern, sentence_lower):
                    # Find the position of the marker with word boundaries
                    marker_match = re.search(pattern, sentence_lower)
                    marker_pos = marker_match.start()
                    
                    # Extract the part of the sentence after the marker
                    text_after_marker = sentence[marker_pos + len(marker):]
                    
                    # Find the first number in this part
                    number_match = re.search(r'-?\d+\.?\d*', text_after_marker)
                    if number_match:
                        conclusion_numbers.append(number_match.group(0))
    
    # Return high priority numbers if found, otherwise return regular conclusion numbers
    return high_priority_numbers if high_priority_numbers else (conclusion_numbers if conclusion_numbers else [])


def extract_answer_from_dataset(text):
   """
   Extracts the answer from the GSM8K dataset examples.

   Args:
       text (str): The dataset example text containing a question and answer.

   Returns:
       str or None: The extracted answer part after the '####' delimiter, or None if not found.

   Explanation:
       1. Checks if the text contains the '####' delimiter that separates question from answer.
       2. If found, splits the text at this delimiter and returns the second part (the answer).
       3. The answer is stripped of leading/trailing whitespace.
       4. Returns None if no delimiter is present.
   """
   if "####" not in text:
       return None
   return text.split("####")[1].strip()


def prepare_dataset(split="train"):
   """
   Load and prepare the GSM8K dataset for training with string prompts.

   Args:
       split (str): The dataset split to load ("train" or "test"). Defaults to "train".

   Returns:
       list: A list of formatted examples, each containing a prompt string and answer.

   Explanation:
       1. Loads the GSM8K dataset from the Hugging Face datasets hub.
       2. For each example in the dataset:
          - Creates a list of messages with system prompt and the question.
          - Converts this list into a single string prompt using build_prompt().
          - Extracts the answer from the dataset example.
          - Creates a formatted example dictionary with prompt and answer.
       3. Returns the list of formatted examples ready for model training or evaluation.
   """
   # GSM8K uses "main" as the configuration name
   data = load_dataset('openai/gsm8k', 'main')[split]
   formatted_data = []
   for example in data:
       # Convert list of messages to a single string prompt.
       prompt_str = build_prompt([
           {"role": "system", "content": SYSTEM_PROMPT},
           {"role": "user", "content": example["question"]}
       ])
       formatted_example = {
           "prompt": prompt_str,  # Now a string rather than a list.
           "answer": extract_answer_from_dataset(example["answer"])
       }
       formatted_data.append(formatted_example)
   return formatted_data

def build_prompt(messages):
   """
   Build a single prompt string from a list of messages.

   Args:
       messages (list): A list of message dictionaries, each with 'role' and 'content' keys.

   Returns:
       str: A concatenated string of all message contents.

   Explanation:
       1. Takes a list of message dictionaries in the typical chat format.
       2. Extracts the 'content' field from each message and strips whitespace.
       3. Joins all content strings with newlines to create a single prompt.
       4. This preserves the training format while converting from structured messages to a string.
   """
   return "\n".join([msg["content"].strip() for msg in messages])


def extract_last_number(text):
   """
   Extracts the last number appearing in the text.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The last number in the text, or None if no number is found.

   Explanation:
       1. Removes dollar signs and percent symbols from the text.
       2. Uses regex to find a number that appears at the end of the text (possibly after whitespace).
       3. The pattern matches numbers that appear at the end of the string, with or without decimal points.
       4. Returns the found number as a float, or None if no match is found.
   """
   text = text.replace('$', '').replace('%', '')
   pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
   match = re.search(pattern, text)
   return float(match.group(1)) if match else None


def extract_single_number(text):
   """
   Extracts a single number from text if exactly one number is present.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The single number in the text, or None if zero or multiple numbers are found.

   Explanation:
       1. Uses regex to find all numbers in the text (including negative numbers and decimals).
       2. If exactly one number is found, returns it as a float.
       3. If zero or multiple numbers are found, returns None.
   """
   numbers = re.findall(r'-?\d*\.?\d+', text)
   return float(numbers[0]) if len(numbers) == 1 else None

def evaluate_model(model, tokenizer, eval_examples, device):
   """
   Evaluates the model on a set of examples and prints detailed results.

   Args:
       model: The language model to evaluate.
       tokenizer: The tokenizer for encoding inputs and decoding outputs.
       eval_examples (list): List of evaluation examples, each containing "prompt" and "answer".
       device: The device (CPU or GPU) to run evaluation on.

   Returns:
       float: The accuracy percentage (correct predictions / total examples * 100).

   Explanation:
       1. Sets the model to evaluation mode.
       2. For each example in the evaluation set:
          - Encodes the prompt and generates a response using the model.
          - Extracts the predicted answer from the generated response.
          - Compares the predicted answer with the expected answer using multiple methods:
            a. Exact string matching
            b. Single number extraction and comparison
            c. Last number extraction and comparison
          - Prints detailed information about each example.
       3. Calculates and returns the overall accuracy.
       4. Returns the model to training mode.
   """
   model.eval()
   correct = 0
   total = len(eval_examples)
   print("\n" + "="*50)
   print("EVALUATION ON", total, "EXAMPLES")
   print("="*50)

   for example in eval_examples:
       # Get the prompt and expected answer
       full_prompt = example["prompt"]
       expected = example["answer"]

       # Tokenize and generate response
       inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
       with torch.no_grad():
           outputs = model.generate(
               inputs,
               max_new_tokens=512,
               temperature=0.7,
               num_return_sequences=1,
               pad_token_id=tokenizer.pad_token_id,
               eos_token_id=tokenizer.eos_token_id,
               forced_eos_token_id=tokenizer.eos_token_id,
               early_stopping=False,
           )
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)

       try:
           # Extract answer and check correctness
           predicted = extract_answer_from_model_output(response)
        #    print("="*50)
        #    print("init predicted: ", predicted)

           # Try different matching methods
           if len(predicted) == 1 and predicted[0] == expected:  # Exact match
               is_correct = True
           elif len(predicted) == 2 and predicted[0] == "..." and predicted[1] == expected:  # Exact match with ellipsis
               is_correct = True
           else:
               # Try single number matching
               pred_num = extract_single_number(str(predicted))
               exp_num = extract_single_number(str(expected))
               if pred_num is not None and exp_num is not None and pred_num == exp_num:
                   is_correct = True
               else:
                   # Try last number matching
                   pred_num = extract_last_number(str(predicted))
                   exp_num = extract_last_number(str(expected))
                   is_correct = (pred_num is not None and exp_num is not None and
                               pred_num == exp_num)

           # Update counter for correct answers
           if is_correct:
               correct += 1

           # Print evaluation details
           print("\nPrompt:")
           print(full_prompt)
           print("\nExpected Answer:")
           print(expected)
           print("\nPredicted Answer:")
           print(predicted)
           print("\nFull Generated Response:")
           print(response)
           print("\nCorrect:", "✓" if is_correct else "✗")
           print("-"*50)

       except Exception as e:
           print("\nFailed to parse model output for prompt:")
           print(full_prompt)
           print("Error:", e)
           print("-"*50)

   # Calculate and print final accuracy
   accuracy = (correct / total) * 100
   print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
   print("="*50)

   # Return model to training mode
   model.train()
   return accuracy