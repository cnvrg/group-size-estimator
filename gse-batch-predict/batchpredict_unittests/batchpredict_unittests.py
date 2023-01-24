import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import shutil
import unittest
import yaml
from batchpredict import ModelNotFoundError
from batchpredict import validate_model_location

np.random.seed(2)


class TestBatchPredict(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Overrides setUpClass from unittest to create artifacts for testing"""
        # Read config file for unittesting
        with open("./test_config.yaml", "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        # Define input arguments for unittesting
        self.dummy_loc = self.config["model_loc"]

        # Run batch predict script on two sample images for unit testing
        os.system(self.config["test_command"])

        # Read count and label files for unit testing
        with open(self.config["run_dir"] + "/counts/bus_counts.txt") as file:
            self.img1_counts = [line.rstrip() for line in file]
        with open(self.config["run_dir"] + "/counts/zidane_counts.txt") as file:
            self.img2_counts = [line.rstrip() for line in file]
        with open(self.config["run_dir"] + "/labels/bus.txt") as file:
            self.img1_labels = [line.rstrip() for line in file]
        with open(self.config["run_dir"] + "/labels/zidane.txt") as file:
            self.img2_labels = [line.rstrip() for line in file]

    def test_model_not_found_error(self):
        """Checks for ModelNotFoundError if the trained model file does not exist at the specified location"""
        with self.assertRaises(ModelNotFoundError):
            validate_model_location(self.dummy_loc)

    def test_output_format(self):
        """Checks if the output files are in the correct format"""
        self.assertTrue(os.path.exists(self.config["run_dir"] + "/bus.jpg"), None)
        self.assertTrue(os.path.exists(self.config["run_dir"] + "/zidane.jpg"), None)
        self.assertTrue(
            os.path.exists(self.config["run_dir"] + "/counts/bus_counts.txt"), None
        )
        self.assertTrue(
            os.path.exists(self.config["run_dir"] + "/counts/zidane_counts.txt"), None
        )
        self.assertTrue(
            os.path.exists(self.config["run_dir"] + "/labels/bus.txt"), None
        )
        self.assertTrue(
            os.path.exists(self.config["run_dir"] + "/labels/zidane.txt"), None
        )

    def test_obj_counts(self):
        """Checks if object counts generated by the model are as expected"""
        self.assertEqual(self.img1_counts, self.config["img1_ref_counts"])
        self.assertEqual(self.img2_counts, self.config["img2_ref_counts"])

    def test_bb_labels(self):
        """Checks if the bounding box coordinates and labels are correct"""
        self.assertAlmostEqual(self.img1_labels, self.config["img1_ref_labels"])
        self.assertAlmostEqual(self.img2_labels, self.config["img2_ref_labels"])

    @classmethod
    def tearDownClass(self):
        """Deletes artiacts generated by unittests"""
        os.remove(self.config["yolo_loc"])
        shutil.rmtree("runs")


if __name__ == "__main__":
    unittest.main()
