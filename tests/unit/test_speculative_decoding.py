import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from src.models.speculative_decoding import SpeculativeDecoder

class TestSpeculativeDecoder(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_target = MagicMock()
        # Ensure model_info exists on target mock
        self.mock_target.model_info = {'vocab_size': 1000}
        
        self.mock_draft = MagicMock() # Removed spec=nn.Module to allow dynamic attributes like generate
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 1
        
        self.device = "cpu"
        
        # Initialize decoder
        self.decoder = SpeculativeDecoder(
            target_model_integrator=self.mock_target,
            draft_model=self.mock_draft,
            draft_k=2,
            tokenizer=self.mock_tokenizer,
            device=self.device
        )

    def test_generate_draft_tokens(self):
        # Setup
        input_ids = torch.tensor([[10, 20]], device=self.device)
        k = 2
        
        # Mock draft model generate return
        # shape: [batch, input_len + k]
        # returns input_ids concatenated with new tokens [100, 101]
        self.mock_draft.generate.return_value = torch.tensor([[10, 20, 100, 101]], device=self.device)
        
        # Action
        draft_tokens = self.decoder._generate_draft_tokens(
            input_ids, k=k, temperature=1.0, top_p=0.9
        )
        
        # Assertions
        self.assertTrue(torch.equal(draft_tokens, torch.tensor([[100, 101]], device=self.device)))
        self.mock_draft.generate.assert_called_once()
        args, kwargs = self.mock_draft.generate.call_args
        self.assertEqual(kwargs['max_new_tokens'], k)
        self.assertEqual(kwargs['do_sample'], True)

    def test_generate_all_accepted(self):
        """Test case where all draft tokens are accepted by target model"""
        prompt_ids = torch.tensor([[1, 2]], device=self.device)
        max_new_tokens = 4
        
        # Mock draft generation: always returns [3, 4] for k=2
        # We need to handle multiple calls if max_new_tokens > k
        # Iteration 1: prompt [1,2] -> draft [3,4]
        # Iteration 2: prompt [1,2,3,4] -> draft [5,6] (not needed if we stop at max_new_tokens)
        
        with patch.object(self.decoder, '_generate_draft_tokens') as mock_draft_gen:
            mock_draft_gen.side_effect = [
                torch.tensor([[3, 4]], device=self.device),
                torch.tensor([[5, 6]], device=self.device)
            ]
            
            # Mock target verification
            # Target needs to validate tokens 3 and 4.
            # Logits shape: [1, seq_len, vocab_size]
            # seq_len for iter 1 is len([1,2,3,4]) = 4.
            # Positions to check are indices -3 (token 2->3) and -2 (token 3->4)?
            # Code says: target_token_id = argmax(target_logits[0, -self.k - 1 + i, :])
            # For i=0 (draft token 0): index = -2 - 1 + 0 = -3. Logits at -3 predicts token at -2 (which is draft[0]?)
            # Usually logits[t] predicts input[t+1].
            # candidate_seq = [1, 2, 3, 4]
            # target runs on [1, 2, 3, 4]
            # output logits for [1, 2, 3, 4]
            # logits[0] -> pred for pos 1 (token 2)
            # logits[1] -> pred for pos 2 (token 3) -> should match draft[0] (3)
            # logits[2] -> pred for pos 3 (token 4) -> should match draft[1] (4)
            # logits[3] -> pred for pos 4 (next)
            
            # Code uses index: -self.k - 1 + i
            # k=2.
            # i=0: -2 - 1 + 0 = -3. Logits at -3 corresponds to prediction for token at index -2.
            # candidate_seq has length 4. Indices: 0, 1, 2, 3.
            # -3 is index 1. Logits[1] predicts token at index 2 (which is 3). Correct.
            # i=1: -2 - 1 + 1 = -2. Logits[-2] is index 2. Logits[2] predicts token at index 3 (which is 4). Correct.
            
            # Construct mock logits that produce 3 and 4 at correct positions
            vocab_size = 1000
            logits_1 = torch.zeros(1, 4, vocab_size, device=self.device)
            logits_1[0, 1, 3] = 10.0 # Predicts 3 at pos 2
            logits_1[0, 2, 4] = 10.0 # Predicts 4 at pos 3
            
            logits_2 = torch.zeros(1, 6, vocab_size, device=self.device)
            # just need valid return
            
            with patch.object(self.decoder, '_run_target_verification') as mock_target_ver:
                mock_target_ver.side_effect = [logits_1, logits_2]
                
                output = self.decoder.generate(prompt_ids, max_new_tokens=2, temperature=0)
                
                # We expect [1, 2, 3, 4]
                self.assertTrue(torch.equal(output, torch.tensor([[1, 2, 3, 4]], device=self.device)))

    def test_generate_rejection(self):
        """Test case where draft token is rejected and corrected"""
        prompt_ids = torch.tensor([[1, 2]], device=self.device)
        
        # k=2
        # Draft generates [3, 4]
        # Target accepts 3, but rejects 4 and predicts 9 instead.
        
        with patch.object(self.decoder, '_generate_draft_tokens') as mock_draft_gen:
            mock_draft_gen.return_value = torch.tensor([[3, 4]], device=self.device)
            
            # Logits setup
            # i=0: index -3 (1). Should predict 3.
            # i=1: index -2 (2). Should predict 9 (NOT 4).
            
            vocab_size = 1000
            logits = torch.zeros(1, 4, vocab_size, device=self.device)
            logits[0, 1, 3] = 10.0 # Accepts 3
            logits[0, 2, 9] = 10.0 # Rejects 4, predicts 9
            
            with patch.object(self.decoder, '_run_target_verification') as mock_target_ver:
                mock_target_ver.return_value = logits
                
                output = self.decoder.generate(prompt_ids, max_new_tokens=2, temperature=0)
                print(f"Output: {output}")
                
                # Expect [1, 2, 3, 9]
                # Logic: 
                # i=0: draft 3, target 3 -> accept.
                # i=1: draft 4, target 9 -> reject. Append target (9). Break.
                # Result: [1, 2, 3, 9]
                self.assertTrue(torch.equal(output, torch.tensor([[1, 2, 3, 9]], device=self.device)))

if __name__ == '__main__':
    unittest.main()
