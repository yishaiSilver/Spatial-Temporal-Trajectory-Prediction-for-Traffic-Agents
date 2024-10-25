
import sys
sys.path.append("../")

import unittest
import numpy as np
import torch

from transformations.agent_center import apply, inverse  # assuming you named the module agent_center

class TestAgentCenter(unittest.TestCase):

    def setUp(self):
        t_in = 3
        t_out = 6
        num_agents = 5

        min_val = 0
        max_val = 1

        # p_in = np.array([
        #     [[0, 0],
        #      [1, 1],
        #      [-1, 5]],
        # ])

        # p_out = np.array([
        #     [[3, 3],
        #      [5, 4],
        #      [3, 5]],
        # ])

        # Create a mock datum for testing
        self.datum = {
            "track_id": np.arange(num_agents),
            "agent_id": 1,
            # "p_in": p_in,
            # "p_out": p_out,
            "p_in": np.random.rand(num_agents, t_in, 2) * max_val + min_val,
            "p_out": np.random.rand(num_agents, t_out, 2) * max_val + min_val,
            "v_in": np.random.rand(num_agents, t_in, 2) * max_val + min_val,
            "v_out": np.random.rand(num_agents, t_out, 2) * max_val + min_val,
            "lane": np.random.rand(100, 2),
            "lane_norm": np.random.rand(100, 2),
        }

    def test_forward_inverse_equals(self):
        agent_index = 1

        target_positions = self.datum["p_out"][agent_index]

        # Test that applying the transformation and then inverting it gives the same result
        transformed_datum = apply(self.datum.copy())

        # print(transformed_datum["p_out"])

        preds = transformed_datum["p_out_transformed"][agent_index]
        preds = torch.tensor(preds, dtype=torch.float32)
        preds = preds.unsqueeze(0) # add batch dim
        # preds: b, t, 2

        print(preds.shape)

        metadata = [transformed_datum["metadata"]]

        preds_inv = inverse(preds, metadata=metadata)
        # b, t, 2

        preds_inv = preds_inv.detach().numpy()
    
        print(target_positions)
        print(preds_inv)

        self.assertTrue(np.allclose(target_positions, preds_inv))

if __name__ == "__main__":
    unittest.main()
