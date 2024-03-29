import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import shutil
import unittest
import yaml
from finetune import DirectoryNotFoundError, ConfigNotFoundError
from finetune import validate_file_locations

np.random.seed(2)


class TestFineTune(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Overrides setUpClass from unittest to create artifacts for testing"""
        # Read config file for unittesting
        with open("./test_config.yaml", "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        # Define input arguments for unittesting
        self.false_img_path = "../images"
        self.false_lbl_path = "../labels"

        # Train model using 4 images for 1 epoch for unit-testing
        os.system(self.config["test_command"])
        self.result_csv = pd.read_csv(self.config["run_dir"] + "/results.csv")

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

    def test_model_exists(self):
        """Checks if the yolov5s.pt file was downloaded correctly from the YOLOv5 GitHub repo"""
        self.assertTrue(os.path.exists(self.config["model_loc"]), None)

    def test_model_metrics(self):
        """Checks if metrics generated by the model are as expected"""
        self.assertAlmostEqual(
            self.result_csv["   metrics/precision"].iloc[0],
            self.config["ref_precision"],
        )
        self.assertAlmostEqual(
            self.result_csv["      metrics/recall"].iloc[0], self.config["ref_recall"]
        )
        self.assertAlmostEqual(
            self.result_csv["     metrics/mAP_0.5"].iloc[0], self.config["ref_map_0.5"]
        )
        self.assertAlmostEqual(
            self.result_csv["metrics/mAP_0.5:0.95"].iloc[0],
            self.config["ref_map_0.5_0.95"],
        )

    @classmethod
    def tearDownClass(self):
        """Deletes artiacts generated by unittests"""
        os.remove(self.config["model_loc"])
        os.remove("./labels/training.cache")
        os.remove("./labels/validation.cache")
        shutil.rmtree("runs")


if __name__ == "__main__":
    unittest.main()
