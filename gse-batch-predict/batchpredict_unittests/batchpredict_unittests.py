import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import yaml
from batchpredict import ModelNotFoundError
from batchpredict import validate_model_location


class TestBatchPredict(unittest.TestCase):
    def setUp(self):
        """Overrides setUp from unittest to create artifacts for testing"""
        # Read config file for unittesting
        with open("./test_config.yaml", "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        # Define input arguments for unittesting
        self.dummy_loc = self.config["model_loc"]


class TestValidateModelLocation(TestBatchPredict):
    def test_model_not_found_error(self):
        """Checks for ModelNotFoundError if the trained model file does not exist at the specified location"""
        with self.assertRaises(ModelNotFoundError):
            validate_model_location(self.dummy_loc)
