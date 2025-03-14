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
        answer (list): List of expected answers.
        device (torch.device, optional): Device to use for GPU operations.
        responses (list, optional): Pre-extracted response content to avoid redundant extraction.
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
    for i in range(0, len(responses), batch_size):
        batch_responses = responses[i:i+batch_size]
        batch_answers = answer[i:i+batch_size]
        
        # Use multiple CPU threads for extraction
        with ThreadPoolExecutor(max_workers=4) as executor:
            extracted_batch = list(executor.map(extract_answer_from_model_output, batch_responses))
        
        for j, (extracted, expected) in enumerate(zip(extracted_batch, batch_answers)):
            idx = i + j
            
            # Handle list format for extracted answers
            if isinstance(extracted, list) and len(extracted) > 0:
                # Check if expected answer is in the list
                if expected in extracted:
                    rewards_tensor[idx] = 2.0
                    continue
                # Otherwise, use the first element for comparison
                extracted = extracted[0]
            
            # Exact match check
            if extracted == expected:
                rewards_tensor[idx] = 2.0
            else:
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
        if "<reasoning>" in response: score += 0.2
        if "</reasoning>" in response: score += 0.2
        if "<answer>" in response: score += 0.2
        if "</answer>" in response: score += 0.2
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