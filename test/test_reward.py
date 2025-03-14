import pytest
import torch
from unittest.mock import patch, MagicMock
from reward import correctness_reward, format_reward, combined_reward

class TestRewardFunctions:
    
    def setup_method(self):
        """Setup common test data"""
        self.cpu_device = torch.device('cpu')
        # Only use CUDA device if available for testing
        self.gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Sample test data
        self.prompts = ["Solve 2+2", "Calculate 5*3"]
        self.completions = [
            [{"content": "<reasoning>Adding 2 and 2</reasoning><answer>4</answer>"}],
            [{"content": "<reasoning>Multiplying 5 and 3</reasoning><answer>15</answer>"}]
        ]
        self.answers = ["4", "15"]
        
        # Completions with varying format compliance
        self.format_completions = [
            [{"content": "<reasoning>Test</reasoning><answer>Test</answer>"}],  # All tags
            [{"content": "<reasoning>Test</reasoning>"}],                      # Missing answer tags
            [{"content": "<answer>Test</answer>"}],                            # Missing reasoning tags
            [{"content": "No tags at all"}]                                    # No tags
        ]
    
    @patch('reward.extract_answer_from_model_output')
    @patch('reward.extract_single_number')
    def test_correctness_reward_exact_match(self, mock_extract_num, mock_extract_answer):
        """Test correctness reward with exact matches"""
        # Setup mocks to return the exact answer
        mock_extract_answer.side_effect = ["4", "15"]
        
        # Call the function
        rewards = correctness_reward(self.prompts, self.completions, self.answers, self.cpu_device)
        
        # Assertions
        assert isinstance(rewards, torch.Tensor)
        assert rewards.device == self.cpu_device
        assert rewards.shape == torch.Size([2])
        assert torch.allclose(rewards, torch.tensor([2.0, 2.0], device=self.cpu_device))
        
        # Verify mocks were called properly
        assert mock_extract_answer.call_count == 2
        assert mock_extract_num.call_count == 0  # Should not be called for exact matches
    
    @patch('reward.extract_answer_from_model_output')
    @patch('reward.extract_single_number')
    def test_correctness_reward_numeric_equivalence(self, mock_extract_num, mock_extract_answer):
        """Test correctness reward with numeric equivalence but not exact match"""
        # Setup mocks for non-exact match but numerically equivalent
        mock_extract_answer.side_effect = ["four", "15.0"]
        
        # Change here: Use a lambda to return values based on input
        # This is more robust than relying on call sequence
        mock_extract_num.side_effect = lambda x: {
            "four": 4,
            "4": 4,
            "15.0": 15,
            "15": 15
        }.get(x, None)
        
        # Call the function
        rewards = correctness_reward(self.prompts, self.completions, self.answers, self.cpu_device)
        
        # Assertions
        assert isinstance(rewards, torch.Tensor)
        assert torch.allclose(rewards, torch.tensor([1.5, 1.5], device=self.cpu_device))
    
    @patch('reward.extract_answer_from_model_output')
    @patch('reward.extract_single_number')
    def test_correctness_reward_incorrect(self, mock_extract_num, mock_extract_answer):
        """Test correctness reward with incorrect answers"""
        # Setup mocks for incorrect answers
        mock_extract_answer.side_effect = ["5", "16"]
        mock_extract_num.side_effect = [5, 4, 16, 15]  # Different values
        
        # Call the function
        rewards = correctness_reward(self.prompts, self.completions, self.answers, self.cpu_device)
        
        # Assertions
        assert isinstance(rewards, torch.Tensor)
        assert torch.allclose(rewards, torch.tensor([0.0, 0.0], device=self.cpu_device))
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch('reward.extract_answer_from_model_output')
    def test_correctness_reward_gpu(self, mock_extract_answer):
        """Test correctness reward on GPU if available"""
        # Setup mock
        mock_extract_answer.side_effect = ["4", "15"]
        
        # Call the function with GPU device
        rewards = correctness_reward(self.prompts, self.completions, self.answers, torch.device('cuda'))
        
        # Assertions
        assert rewards.device.type == 'cuda'
        assert torch.allclose(rewards, torch.tensor([2.0, 2.0], device=torch.device('cuda')))
    
    def test_format_reward(self):
        """Test format reward calculation"""
        # Calculate format scores
        rewards = format_reward(self.format_completions, self.cpu_device)
        
        # Expected scores: 
        # - 0.8 for all tags
        # - 0.4 for missing answer tags
        # - 0.4 for missing reasoning tags  
        # - 0.0 for no tags
        expected = torch.tensor([0.8, 0.4, 0.4, 0.0], device=self.cpu_device)
        
        # Assertions
        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == torch.Size([4])
        assert torch.allclose(rewards, expected)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_format_reward_gpu(self):
        """Test format reward on GPU if available"""
        # Calculate format scores on GPU
        rewards = format_reward(self.format_completions, torch.device('cuda'))
        
        # Assertions
        assert rewards.device.type == 'cuda'
        assert rewards.shape == torch.Size([4])
    
    @patch('reward.correctness_reward')
    @patch('reward.format_reward')
    def test_combined_reward(self, mock_format, mock_correctness):
        """Test combined reward calculation"""
        # Setup mocks
        mock_correctness.return_value = torch.tensor([2.0, 1.5], device=self.cpu_device)
        mock_format.return_value = torch.tensor([0.8, 0.4], device=self.cpu_device)
        
        # Call combined reward
        combined = combined_reward(self.prompts, self.completions, self.answers, self.cpu_device)
        
        # Assertions
        assert isinstance(combined, torch.Tensor)
        assert torch.allclose(combined, torch.tensor([2.8, 1.9], device=self.cpu_device))
        
        # Verify mocks were called
        mock_correctness.assert_called_once()
        mock_format.assert_called_once()
    
    @patch('reward.correctness_reward')
    @patch('reward.format_reward')
    def test_combined_reward_default_device(self, mock_format, mock_correctness):
        """Test combined reward with default device"""
        # Setup device-aware mocks
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mock_correctness.return_value = torch.tensor([2.0, 1.5], device=default_device)
        mock_format.return_value = torch.tensor([0.8, 0.4], device=default_device)
        
        # Call without specifying device
        combined = combined_reward(self.prompts, self.completions, self.answers)
        
        # Assertions
        assert combined.device.type == default_device.type
        assert torch.allclose(combined, torch.tensor([2.8, 1.9], device=default_device))


class TestIntegrationWithExtractFunction:
    """Integration tests with real extract functions"""
    
    @pytest.fixture
    def mock_extract_functions(self):
        """Create fixture with mock extraction functions"""
        with patch('reward.extract_answer_from_model_output') as mock_extract:
            with patch('reward.extract_single_number') as mock_extract_num:
                yield mock_extract, mock_extract_num
    
    def test_correctness_with_boxed_answer(self, mock_extract_functions):
        """Test with LaTeX boxed answers"""
        mock_extract, mock_extract_num = mock_extract_functions
        
        # Setup for boxed format
        mock_extract.side_effect = [["42"], ["3.14"]]
        
        prompts = ["Problem 1", "Problem 2"]
        completions = [
            [{"content": "The answer is \\boxed{42}"}],
            [{"content": "The answer is \\boxed{3.14}"}]
        ]
        answers = ["42", "3.14"]
        
        # Call the function
        rewards = correctness_reward(prompts, completions, answers)
        
        # Assertions - should match first element in list
        assert torch.allclose(rewards, torch.tensor([2.0, 2.0], device=rewards.device))
    
    def test_format_reward_real_responses(self):
        """Test format reward with real-world response patterns"""
        completions = [
            [{"content": "<reasoning>\nStep 1: Calculate stuff\nStep 2: More math\n</reasoning>\n<answer>42</answer>"}],
            [{"content": "I think the answer is 15."}],
            [{"content": "<answer>3.14</answer>\n<reasoning>This is out of order</reasoning>"}],
            [{"content": "<reasoning>Partial tags only"}]
        ]
        
        rewards = format_reward(completions)
        
        # Expected scores based on tag presence
        expected = torch.tensor([0.8, 0.0, 0.8, 0.2], device=rewards.device)
        assert torch.allclose(rewards, expected)

    def test_end_to_end_reward_calculation(self):
        """End-to-end test of the entire reward pipeline with real examples"""
        prompts = ["Calculate 2+2", "Find the area of a circle with radius 2"]
        completions = [
            [{"content": "<reasoning>To calculate 2+2, I add the numbers directly.\n2+2=4</reasoning>\n<answer>4</answer>"}],
            [{"content": "The area of a circle is π×r². For r=2, area = π×2² = 4π ≈ 12.57\n\nTherefore, \\boxed{12.57}"}]
        ]
        answers = ["4", "12.57"]
        
        # Calculate rewards
        correctness = correctness_reward(prompts, completions, answers)
        format_scores = format_reward(completions)
        combined = combined_reward(prompts, completions, answers)
        
        # Check that combined equals the sum of individual rewards
        assert torch.allclose(combined, correctness + format_scores)
        
        # Check value ranges
        assert torch.all(correctness >= 0) and torch.all(correctness <= 2.0)
        assert torch.all(format_scores >= 0) and torch.all(format_scores <= 0.8)
        assert torch.all(combined >= 0) and torch.all(combined <= 2.8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_large_batch_processing(self):
        """Test with a larger batch to verify batch processing works correctly"""
        # Create a larger batch
        n_samples = 40  # Larger than our batch size of 16
        prompts = [f"Problem {i}" for i in range(n_samples)]
        completions = [[{"content": f"<reasoning>Calculation {i}</reasoning><answer>{i}</answer>"}] for i in range(n_samples)]
        answers = [str(i) for i in range(n_samples)]
        
        # Process on GPU if available
        device = torch.device("cuda")
        rewards = correctness_reward(prompts, completions, answers, device)
        
        # All should have perfect scores since answers match
        assert rewards.shape[0] == n_samples
        assert torch.all(rewards == 2.0)
        assert rewards.device.type == "cuda"