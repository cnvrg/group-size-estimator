import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import yaml
from finetune import DirectoryNotFoundError, ConfigNotFoundError
from finetune import validate_file_locations


class TestFineTune(unittest.TestCase):
    def setUp(self):
        """Overrides setUp from unittest to create artifacts for testing"""
        # Read config file for unittesting
        with open("./test_config.yaml", "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        # Define input arguments for unittesting
        self.false_img_path = "./images"
        self.false_lbl_path = "./labels"


class TestValidateFileLocations(TestFineTune):
    def test_dir_not_found_error(self):
        """Checks for DirectoryNotFoundError if the image or label directory does not exist"""
        with self.assertRaises(DirectoryNotFoundError):
            validate_file_locations(
                self.false_img_path,
                self.config["labels_loc"],
                self.config["config_loc"],
            )
        with self.assertRaises(DirectoryNotFoundError):
            validate_file_locations(
                self.config["images_loc"],
                self.false_lbl_path,
                self.config["config_loc"],
            )

    def test_config_not_found_error(self):
        """Checks for ConfigNotFoundError if the dataset config file does not exist"""
        with self.assertRaises(ConfigNotFoundError):
            validate_file_locations(
                self.config["images_loc"],
                self.config["labels_loc"],
                self.config["config_loc"],
            )
