###########################
# Step 4. LOAD AND TEST MODEL  #
###########################
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataset_process import evaluate_model, prepare_dataset
import random
from dataset_process import SYSTEM_PROMPT, build_prompt, extract_answer_from_model_output


def evaluate_model_on_specific_question(model, tokenizer, prompts_to_test, device):
    """
    Evaluates the model on a specific question.

    Args:
        model: The fine-tuned model.
        tokenizer: The tokenizer for the model.
        question: The question to evaluate the model on.
        device: The device to run the evaluation on.

    Returns:
        The accuracy of the model on the question.
    """
    # Define test prompts
    prompts_to_test = [
        # "How much is 1+1?",
        # "I have 3 apples, my friend eats one and I give 2 to my sister, how many apples do I have now?",
        # "Solve the equation 6x + 4 = 40"
        # "Solve the equation 8x + 10 = 20"
        "Which number is bigger? 3.11 v.s. 3.9"
    ]

    # Test each prompt
    for prompt in prompts_to_test:
        print("================================================================")
        # Prepare the prompt using the same format as during training
        test_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        test_prompt = build_prompt(test_messages)

        # Tokenize the prompt and generate a response
        test_input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)

        # Generate response with similar parameters to those used in training
        with torch.no_grad():
            test_output_ids = model.generate(
                test_input_ids,
                max_new_tokens=400,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                early_stopping=False
            )

        test_response = tokenizer.decode(test_output_ids[0], skip_special_tokens=True)

        # Print the test prompt and the model's response
        print("\nTest Prompt:")
        print(test_prompt)
        print("\nModel Response:")
        print(test_response)

        # Extract and display the answer part for easier evaluation
        try:
            extracted_answer = extract_answer_from_model_output(test_response)
            print("\nExtracted Answer:")
            print(extracted_answer)
            print("-" * 50)
        except Exception as e:
            print(f"\nFailed to extract answer: {e}")
            print("-" * 50)


def evaluate_model_on_dataset(model, tokenizer, device):
    """
    Evaluates the model on a dataset of questions.

    Args:
        model: The fine-tuned model.
        tokenizer: The tokenizer for the model.
        dataset: The dataset of questions to evaluate the model on.
        device: The device to run the evaluation on.

    Returns:
        The accuracy of the model on the dataset.
    """
    # Load training data
    train_data = prepare_dataset("train")
    random.shuffle(train_data)

    # Load validation data separately instead of splitting training data
    validation_data = prepare_dataset("test")  # GSM8K uses "test" as the validation split name
    print(f"Loaded {len(train_data)} training examples and {len(validation_data)} validation examples")

    # Optional: Use a smaller subset of validation data for quicker evaluation during development
    size_of_eval_data = min(10, len(validation_data))
    # eval_data = validation_data[:size_of_eval_data]
    eval_data = train_data[:size_of_eval_data]


    print("\nFinal model evaluation on validation data:")
    post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")



def main():
    """
    Main function to load the fine-tuned model and test it on example math problems.

    Explanation:
        1. Determines the device (GPU if available, otherwise CPU).
        2. Loads the fine-tuned model and tokenizer from the saved path.
        3. Tests the model on predefined math problems.
        4. Formats the prompt using the same SYSTEM_PROMPT and build_prompt function as training.
        5. Generates and displays responses for each test prompt.
    """
    # Determine the device: use GPU if available, else fallback to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the saved model and tokenizer
    saved_model_path = "grpo_finetuned_model"
    # saved_model_path = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    # saved_model_path = "Qwen/Qwen2.5-1.5B-Instruct"


    # Load the model
    loaded_model = AutoModelForCausalLM.from_pretrained(
        saved_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )


    loaded_tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
    loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
    loaded_model.config.pad_token_id = loaded_tokenizer.eos_token_id
    loaded_model.config.eos_token_id = loaded_tokenizer.eos_token_id
    
    evaluate_model_on_dataset(loaded_model, loaded_tokenizer, device)


if __name__ == "__main__":
    main()
