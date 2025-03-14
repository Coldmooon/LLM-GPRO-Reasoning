import pytest
from dataset_process import extract_answer_from_model_output

def test_extract_function_debug():
    """Test the extract_answer_from_model_output function with real data for debugging."""
    # Load content from test_text.txt
    with open('test_text.txt', 'r') as file:
        test_content = file.read()
    
    # Extract answer using the function
    result = extract_answer_from_model_output(test_content)
    
    # Print the result in a formatted way for debugging
    print("\n" + "="*50)
    print("EXTRACT FUNCTION DEBUG OUTPUT")
    print("="*50)
    print(f"Input text length: {len(test_content)} characters")
    print("-"*50)
    print("Extracted result:")
    
    if result is None:
        print("No answer extracted (None returned)")
    elif isinstance(result, list):
        print(f"List of {len(result)} answers:")
        for i, answer in enumerate(result, 1):
            print(f"  {i}. '{answer}'")
    else:
        print(f"Single answer: '{result}'")
    
    print("="*50)
    
    # Return the result for potential assertions
    return result

if __name__ == "__main__":
    # Run the test directly when this file is executed
    test_extract_function_debug()
