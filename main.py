# Import necessary libraries
# Basic Python libraries for various operations
import random
import os
import numpy as np
import wandb

# PyTorch and related libraries for deep learning
import torch
from torch.nn.utils.rnn import pad_sequence

# Hugging Face libraries for transformer models
from transformers import AutoModelForCausalLM, AutoTokenizer


from dataset_process import evaluate_model, prepare_dataset
from reward import combined_reward
from GRPO import train_with_grpo

def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None

    Explanation:
        1. Sets seed for Python's built-in random module for basic random operations.
        2. Sets seed for NumPy, ensuring consistent random number generation in array operations.
        3. Sets seed for PyTorch CPU operations.
        4. If CUDA is available, sets seed for all GPU devices.
        5. Configures cuDNN to ensure deterministic behavior:
           - Sets deterministic flag to True, ensuring reproducible results.
           - Disables benchmarking to prevent algorithm selection based on hardware.

    Note:
        Setting deterministic behavior may impact performance but ensures consistent results
        across multiple runs, which is crucial for debugging and research.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call the function to set random seed for reproducibility
set_random_seed(42)

# Set environment variables for Weights & Biases (wandb) logging
os.environ["WANDB_API_KEY"] = "4cfa2def1589486b10cbbbd1f8eb710393151359"
os.environ["WANDB_PROJECT"] = "GRPO-Qwen-1.5-Instruct-Multi-GPU"


def optimize_model_memory(model):
    """
    Optimizes the model to use less memory during training.

    Args:
        model: The language model to optimize.

    Returns:
        The optimized model.

    Explanation:
        1. Sets the model to training mode.
        2. Disables KV caching to save memory.
        3. Enables gradient checkpointing to trade computation for memory.
        4. Ensures that input embeddings require gradients:
           - Either uses the built-in method if available.
           - Or adds a forward hook to the input embeddings layer.
        5. Returns the optimized model ready for memory-efficient training.
    """
    model.train()
    model.config.use_cache = False

    # First ensure inputs will require gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model

# Main execution
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using primary device: {device}")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir = "math_solver_model"

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Model downloaded")

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

num_gpus = torch.cuda.device_count()
print(f"Detected {num_gpus} GPUs")
device_ids = list(range(num_gpus)) if num_gpus > 1 else None

# Load training data
train_data = prepare_dataset("train")
random.shuffle(train_data)

# Load validation data separately instead of splitting training data
validation_data = prepare_dataset("test")  # GSM8K uses "test" as the validation split name
print(f"Loaded {len(train_data)} training examples and {len(validation_data)} validation examples")

# Optional: Use a smaller subset of validation data for quicker evaluation during development
size_of_eval_data = min(30, len(validation_data))

# final data
train_data = train_data[size_of_eval_data:]
eval_data = train_data[:size_of_eval_data]

# print("\nInitial model evaluation before finetuning:")
# pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
# print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

model = optimize_model_memory(model)

print("\nStarting RL fine-tuning using GRPO...")
# ***
# This config was tested on a 8xA100 node, where each A100 is has 80GB of VRAM
# 重点： 默认配置是给 8 卡 A100的，所有 batch size 设置为 7
# ***
training_config = {
    'num_iterations': 1,
    'num_steps': 1164, # ,,
    'batch_size': 3, # 7 #reduce if you have fewer GPUs
    'num_generations': 9, # 12, # reduce if you have GPUs with less VRAM
    'max_completion_length': 400, #400, # reduce if you have GPUs with less VRAM
    'beta': 0.04,
    'learning_rate': 5e-6,
    'mu': 1,
    'epsilon': 0.1
}

# Initialize Weights & Biases
wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True)
print("Weights & Biases initialized.")

model = train_with_grpo(
    model=model,
    tokenizer=tokenizer,
    train_data=train_data,
    reward_function=combined_reward,
    device_ids=device_ids,
    **training_config
)

wandb.finish()
print("Training completed and wandb run finished.")

print("\nFinal model evaluation on validation data:")
post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")

# Optional: Evaluate on the full validation set
# if len(validation_data) > size_of_eval_data:
    # print("\nEvaluating on full validation set:")
    # full_validation_accuracy = evaluate_model(model, tokenizer, validation_data, device)
    # print(f"Full Validation Accuracy: {full_validation_accuracy:.2f}%")

print("\nSaving GRPO fine-tuned model...")
model.save_pretrained("grpo_finetuned_model")
tokenizer.save_pretrained("grpo_finetuned_model")
