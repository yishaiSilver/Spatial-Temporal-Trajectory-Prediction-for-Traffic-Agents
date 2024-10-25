"""
This is a test for the agent_center transformation.
"""

import sys

import unittest
import numpy as np
import torch

sys.path.append("../")

from transformations.agent_center import apply, inverse
from utils.logger_config import logger


class TestAgentCenter(unittest.TestCase):
    """
    Test the agent_center transformation.
    """

    def setUp(self):
        """
        Set up the test case.
        """

        t_in = 3
        t_out = 6
        num_agents = 5

        min_val = 0
        max_val = 10

        # Create a mock datum for testing
        self.datum = {
            "track_id": np.arange(num_agents),
            "agent_id": np.random.randint(0, num_agents),
            "p_in": np.random.rand(num_agents, t_in, 2) * max_val + min_val,
            "p_out": np.random.rand(num_agents, t_out, 2) * max_val + min_val,
            "v_in": np.random.rand(num_agents, t_in, 2) * max_val + min_val,
            "v_out": np.random.rand(num_agents, t_out, 2) * max_val + min_val,
            "lane": np.random.rand(100, 2),
            "lane_norm": np.random.rand(100, 2),
        }

    def test_forward_inverse_equals(self):
        """
        Test that forward and inverse transformations are equal.
        """

        agent_index = self.datum["agent_id"]

        target_positions = self.datum["p_out"][agent_index]

        # Test that applying the transformation and then inverting it gives the same result
        transformed_datum = apply(self.datum.copy())

        # preds: b, t, 2
        preds = transformed_datum["p_out_transformed"][agent_index]
        preds = torch.tensor(preds, dtype=torch.float32)
        preds = preds.unsqueeze(0)  # add batch dim

        metadata = [transformed_datum["metadata"]]

        # preds_inv: b, t, 2
        preds_inv = inverse(preds, metadata=metadata)

        preds_inv = preds_inv.detach().numpy()

        logger.debug("\n%s", target_positions)
        logger.debug("\n%s", preds_inv)

        self.assertTrue(np.allclose(target_positions, preds_inv))


if __name__ == "__main__":
    unittest.main()
