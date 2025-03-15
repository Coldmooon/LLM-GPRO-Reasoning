from dataset_process import extract_answer_from_model_output, extract_single_number, extract_last_number
import re
import torch
from concurrent.futures import ThreadPoolExecutor


def extract_response_content(completions):
    """Helper function to extract content from completions once."""
    return [completion[0]['content'] for completion in completions]


def create_reward_tensor(size, device=None):
    """Helper function to create a zero-initialized reward tensor on the specified device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.zeros(size, device=device)


def correctness_reward(prompts, completions, answer, device=None, responses=None, reward_tensor=None, **kwargs):
    """
    Assigns a reward based on the correctness of the model's answer.
    
    Args:
        prompts (list): List of input prompts.
        completions (list): List of model completions, each containing content.
        answer (list, size [27]): List of expected answers. 
        device (torch.device, optional): Device to use for GPU operations.
        responses (list, optional, size [27]): Pre-extracted response content to avoid redundant extraction.
        reward_tensor (torch.Tensor, optional): Pre-allocated tensor for rewards.
        **kwargs: Additional keyword arguments.
        
    Returns:
        torch.Tensor: Tensor of numerical rewards for each completion.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract content if not provided
    if responses is None:
        responses = extract_response_content(completions)
    
    # Use pre-allocated tensor or create a new one
    if reward_tensor is None:
        rewards_tensor = create_reward_tensor(len(responses), device)
    else:
        rewards_tensor = reward_tensor
    
    # Process in batches
    batch_size = 16
    # # Debug information about input sizes
    # print(f"\nDebug Information:")
    # print(f"Number of responses: {len(responses)}")
    # print(f"Number of answers: {len(answer)}")
    # print(f"answer：{answer}")
    # # Validate input lengths match
    # if len(responses) != len(answer):
    #     raise ValueError(f"Mismatch between responses ({len(responses)}) and answers ({len(answer)})")
    
    # # Sample debugging of first few items
    # debug_samples = min(3, len(responses))
    # print(f"\nSampling first {debug_samples} items for detailed inspection:")
    # for i in range(debug_samples):
    #     print(f"\nItem {i+1}:")
    #     print(f"Response: {responses[i][:200]}...") # Show first 200 chars
    #     print(f"Expected Answer: {answer[i]}")
    #     extracted = extract_answer_from_model_output(responses[i])
    #     print(f"Extracted Answer: {extracted}")
        
    #     # Validate extracted answer format
    #     if isinstance(extracted, list):
    #         print(f"Note: Extracted answer is a list with {len(extracted)} elements")
    #     elif extracted is None:
    #         print("Warning: No answer could be extracted from response")
    for i in range(0, len(responses), batch_size):
        batch_responses = responses[i:i+batch_size]
        batch_answers = answer[i:i+batch_size]
        
        # Use multiple CPU threads for extraction
        with ThreadPoolExecutor(max_workers=4) as executor:
            extracted_batch = list(executor.map(extract_answer_from_model_output, batch_responses))
        
        for j, (extracted, expected) in enumerate(zip(extracted_batch, batch_answers)):
            idx = i + j
            
            # Check if extracted is None
            if extracted is None:
                # Handle the case where no answer was extracted
                rewards_tensor[idx] = 0.0
                continue
                
            # Exact match check
            if len(extracted) == 1 and extracted[0] == expected:
                rewards_tensor[idx] = 2.0
                continue
            # part of the list matched
            elif isinstance(extracted, list) and len(extracted) > 0:
                # Check if expected answer is in the list
                if expected in extracted:
                    rewards_tensor[idx] = 1.0
                    continue
            # Otherwise, use the first element for comparison
            # Try numeric equivalence
            r_num = extract_single_number(str(extracted))
            a_num = extract_single_number(str(expected))
            
            if r_num is not None and a_num is not None and r_num == a_num:
                rewards_tensor[idx] = 1.5
    
    return rewards_tensor


def format_reward(completions, device=None, responses=None, reward_tensor=None, **kwargs):
    """
    Assigns a reward for adhering to the desired XML format.
    
    Args:
        completions (list): List of model completions, each containing content.
        device (torch.device, optional): Device to use for GPU operations.
        responses (list, optional): Pre-extracted response content to avoid redundant extraction.
        reward_tensor (torch.Tensor, optional): Pre-allocated tensor for rewards.
        **kwargs: Additional keyword arguments.
        
    Returns:
        torch.Tensor: Tensor of format compliance scores for each completion.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract content if not provided
    if responses is None:
        responses = extract_response_content(completions)
    
    # Use pre-allocated tensor or create a new one
    if reward_tensor is None:
        rewards_tensor = create_reward_tensor(len(responses), device)
    else:
        rewards_tensor = reward_tensor
    
    # Calculate format scores
    for idx, response in enumerate(responses):
        score = 0.0
        # if "<reasoning>" in response: score += 0.2
        # if "</reasoning>" in response: score += 0.2
        # if "<answer>" in response: score += 0.2

        # Penalize for multiple tag pairs
        reasoning_starts = response.count("<reasoning>")
        reasoning_ends = response.count("</reasoning>")
        answer_starts = response.count("<answer>")
        answer_ends = response.count("</answer>")
        
        if reasoning_starts == 1:
            score += 0.2
        if reasoning_ends == 1:
            score += 0.2
        if answer_starts == 1:
            score += 0.2
        if answer_ends == 1:    
            score += 0.2


        if reasoning_starts > 1 or reasoning_ends > 1:
            score += 0.1
            print(f"Warning: Found multiple reasoning tags")
        if answer_starts > 1 or answer_ends > 1:
            score += 0.1
            print(f"Warning: Found multiple answer tags")

        if reasoning_starts != reasoning_ends:
            score -= 0.2
            print(f"Warning: Found mismatched reasoning tags")
        if answer_starts != answer_ends:
            score -= 0.2    
            print(f"Warning: Found mismatched answer tags")

        # Penalize if there's text after the final </answer> tag
        if answer_ends >= 1:
            last_tag_pos = response.rindex("</answer>")
            remaining_text = response[last_tag_pos + len("</answer>"):].strip()
            if remaining_text:
                score -= 0.2
                # print(f"Warning: Found text after final </answer> tag: {remaining_text}")   
        rewards_tensor[idx] = score
    
    return rewards_tensor


def combined_reward(prompts, completions, answer, device=None):
    """
    Combines correctness and format rewards efficiently.
    
    Args:
        prompts (list[str]): List of prompt texts
        completions (list[list[dict]]): List of completion dictionaries
        answer (list[str]): List of expected answers
        device (torch.device, optional): Device to use for GPU operations.
        
    Returns:
        torch.Tensor: Combined rewards for each prompt-completion pair
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract responses once
    responses = extract_response_content(completions)
    
    # Create a single rewards tensor to store the combined rewards
    final_rewards = create_reward_tensor(len(responses), device)
    
    # Calculate correctness rewards directly into final_rewards tensor
    correctness_scores = correctness_reward(
        prompts, completions, answer, device, 
        responses=responses, reward_tensor=final_rewards.clone()
    )
    
    # Calculate format rewards in a separate tensor
    format_scores = format_reward(
        completions, device, responses=responses
    )
    
    # Add format scores to the correctness scores
    final_rewards = correctness_scores + format_scores
    
    return final_rewards