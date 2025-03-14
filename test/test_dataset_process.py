import pytest
from dataset_process import (
    extract_answer_from_model_output,
    extract_from_answer_tags,
    extract_from_latex_boxed,
    extract_from_conclusion_sentences
)

class TestExtractAnswerFromModelOutput:
    
    def test_extract_from_answer_tags(self):
        """Test extracting answers from <answer> tags"""
        # Single answer tag
        text = "<reasoning>Some reasoning here</reasoning>\n<answer>42</answer>"
        result = extract_answer_from_model_output(text)
        assert result == ["42"]
        
        # Multiple answer tags
        text = "<answer>First answer</answer>\nSome text\n<answer>42</answer>"
        result = extract_answer_from_model_output(text)
        assert result == ["42"]  # Should return the last answer
        
        # Answer tag with '...'
        text = "<answer>\n...\n</answer>\n<answer>42</answer>"
        result = extract_answer_from_model_output(text)
        assert result == ["42"]  # Should ignore '...'
    
    def test_extract_from_latex_boxed(self):
        """Test extracting answers from LaTeX boxed format"""
        # Single boxed answer
        text = "The solution is \\boxed{42}"
        result = extract_answer_from_model_output(text)
        assert result == ["42"]
        
        # Multiple boxed answers
        text = "First attempt: \\boxed{24}\nFinal answer: \\boxed{42}"
        result = extract_answer_from_model_output(text)
        assert result == ["24", "42"]  # Should return all boxed answers
        
        # Boxed answer with spaces
        text = "The answer is \\boxed{ 42 }"
        result = extract_answer_from_model_output(text)
        assert result == ["42"]
        
        # Complex boxed content
        text = "\\boxed{x = 10 \\text{ meters}}"
        result = extract_answer_from_model_output(text)
        assert result == ["x = 10 \\text{ meters}"]
    
    def test_extract_from_conclusion_sentences(self):
        """Test extracting answers from conclusion sentences"""
        # High priority markers
        text = "The calculation shows 12. The final answer is 42."
        result = extract_answer_from_model_output(text)
        assert result == ["42"]
        
        # Regular conclusion markers
        text = "We add 2 and 3. Therefore 5 is our answer."
        result = extract_answer_from_model_output(text)
        assert result == ["5"]
        
        # Multiple conclusion markers
        text = "Therefore 12 is the value. Thus, 42 is the final result."
        result = extract_answer_from_model_output(text)
        assert result == ["12", "42"]
    
    def test_extraction_priority(self):
        """Test that extraction methods follow the correct priority order"""
        # Should use answer tags first
        text = "<answer>42</answer>\n\\boxed{24}\nTherefore 15."
        result = extract_answer_from_model_output(text)
        assert result == ["42"]
        
        # Should use boxed answers if no answer tags
        text = "\\boxed{42}\nTherefore 15."
        result = extract_answer_from_model_output(text)
        assert result == ["42"]
        
        # Should use conclusion sentences if no answer tags or boxed answers
        text = "Therefore, 42 is the answer."
        result = extract_answer_from_model_output(text)
        assert result == ["42"]
    
    def test_no_extraction(self):
        """Test when no answers can be extracted"""
        text = "This text doesn't contain any extractable answers."
        result = extract_answer_from_model_output(text)
        assert result is None
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Empty text
        assert extract_answer_from_model_output("") is None
        
        # Malformed tags
        text = "<answer>42"  # Missing closing tag
        result = extract_answer_from_model_output(text)
        assert result == []
        
        # Invalid boxed syntax
        text = "\\boxed42}"  # Missing opening brace
        result = extract_answer_from_model_output(text)
        assert result == []


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", "test_dataset_process.py"]) 