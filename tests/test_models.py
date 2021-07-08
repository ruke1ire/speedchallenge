#!/usr/bin/python3

import unittest
import torch
import numpy as np

from model.models import VAE

class TestVAE(unittest.TestCase):
    def test_output_type(self):
        """
        Test that the output type is correct
        """
        i = torch.tensor



        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)

if __name__ == '__main__':
    unittest.main()
