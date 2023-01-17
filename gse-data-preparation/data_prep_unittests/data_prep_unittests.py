import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest
import yaml
from data_preparation import (
    NoneDatasetError,
    DatasetPathError,
    ClassFileFormatError,
    ValidationSizeError,
    DatasetSizeError,
    DatasetNamingError,
    NumberOfClassesError,
)
from data_preparation import validate_arguments, validate_dataset, train_valid_split

np.random.seed(2)


class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        """Overrides setUp from unittest to create artifacts for testing"""
        # Read config file for unittesting
        with open("./test_config.yaml", "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        # Define paths from config file for validate_arguments function
        self.dataset = config["dataset_dir"]
        self.data_dir = config["dataset_full"]
        self.img_dir = config["dataset_images"]
        self.lbl_dir = config["dataset_labels"]
        self.class_txt = config["classes_txt"]
        self.class_csv = config["classes_csv"]

        # Define input arguments and expected answers for validate_dataset function
        self.img_formats = ["jpg"]
        self.data_list = [
            "1.jpg",
            "1.txt",
            "2.jpg",
            "2.txt",
            "3.jpg",
            "3.txt",
            "4.jpg",
            "4.txt",
            "5.jpg",
            "5.txt",
        ]
        self.img_list = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]
        self.lbl_list = ["1.txt", "2.txt", "3.txt", "4.txt", "5.txt"]
        self.class_list = ["oranges", "apples", "bananas"]
        self.num_images = 5
        self.num_labels = 5
        self.num_classes = 3
        self.img_list_v1 = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]
        self.lbl_list_v1 = ["1.txt", "2.txt", "3.txt", "4.txt"]
        self.lbl_list_v2 = ["1.txt", "2.txt", "3.txt", "4.txt", "6.txt"]
        self.class_list_v1 = []

        # Define input arguments and expected answers for train_valid_split function
        self.valid_size = 0.4
        self.num_train = 3
        self.num_valid = 2
        self.train_imgs = ["3.jpg", "1.jpg", "4.jpg"]
        self.valid_imgs = ["2.jpg", "5.jpg"]
        self.train_lbls = ["3.txt", "1.txt", "4.txt"]
        self.valid_lbls = ["2.txt", "5.txt"]


class TestValidateArguments(TestDataPreparation):
    def test_none_dataset_error(self):
        """Checks for NoneDatasetError if all directories are None"""
        with self.assertRaises(NoneDatasetError):
            validate_arguments("None", "None", "None", "classes.txt", "0.1")

    def test_dataset_path_error(self):
        """Checks for DatasetPathError if the images directory is None and labels directory is not or vice-versa"""
        with self.assertRaises(DatasetPathError):
            validate_arguments("None", self.img_dir, "None", "classes.txt", "0.1")
        with self.assertRaises(DatasetPathError):
            validate_arguments("None", "None", self.lbl_dir, "classes.txt", "0.1")

    def test_class_file_format_error(self):
        """Checks for ClassFileFormatError if classes file is not in txt or csv format"""
        with self.assertRaises(ClassFileFormatError):
            validate_arguments("None", self.img_dir, self.lbl_dir, "classes.jpg", "0.1")

    def test_validation_size_error(self):
        """Checks for ValidationSizeError if validation size is not between 0 and 0.4"""
        with self.assertRaises(ValidationSizeError):
            validate_arguments(
                "None", self.img_dir, self.lbl_dir, self.class_txt, "0.6"
            )


class TestValidateDataset(TestDataPreparation):
    def test_data_dir_input(self):
        """Checks for appropriate output when both images and labels are present in one directory"""
        result = validate_dataset(
            self.data_list, [], [], self.img_formats, self.class_list
        )
        self.assertIsInstance(result[0], list)
        self.assertIsInstance(result[1], list)
        self.assertEqual(
            [len(result[0]), len(result[1])], [self.num_images, self.num_labels]
        )

    def test_img_lbl_input(self):
        """Checks for appropriate output when images and labels are present in different directories"""
        result = validate_dataset(
            [], self.img_list, self.lbl_list, self.img_formats, self.class_list
        )
        self.assertIsInstance(result[0], list)
        self.assertIsInstance(result[1], list)
        self.assertEqual(
            [len(result[0]), len(result[1])], [self.num_images, self.num_labels]
        )

    def test_dataset_size_error(self):
        """Checks for DatasetSizeError if number of images is not equal to number of labels"""
        with self.assertRaises(DatasetSizeError):
            validate_dataset(
                [],
                self.img_list_v1,
                self.lbl_list_v1,
                self.img_formats,
                self.class_list,
            )

    def test_dataset_naming_error(self):
        """Checks for DatasetNamingError if an image does not have a corresponding label"""
        with self.assertRaises(DatasetNamingError):
            validate_dataset(
                [],
                self.img_list_v1,
                self.lbl_list_v2,
                self.img_formats,
                self.class_list,
            )

    def test_number_of_classes_error(self):
        """Checks for NumberOfClassesError if no classes/categories have been defined in classes file"""
        with self.assertRaises(NumberOfClassesError):
            validate_dataset(
                [], self.img_list, self.lbl_list, self.img_formats, self.class_list_v1
            )


class TrainValidSplit(TestDataPreparation):
    def test_train_valid_sets(self):
        """Checks datatype and sizes of train and validation sets"""
        train_imgs, val_imgs, train_lbls, val_lbls = train_valid_split(
            self.img_list, self.lbl_list, self.valid_size
        )
        self.assertIsInstance(train_imgs, list)
        self.assertIsInstance(val_imgs, list)
        self.assertIsInstance(train_lbls, list)
        self.assertIsInstance(val_lbls, list)
        self.assertEqual(
            [len(train_imgs), len(val_imgs), len(train_lbls), len(val_lbls)],
            [self.num_train, self.num_valid, self.num_train, self.num_valid],
        )

    def test_exact_output(self):
        """Checks if the training and validation sets are generated as expected"""
        train_imgs, val_imgs, train_lbls, val_lbls = train_valid_split(
            self.img_list, self.lbl_list, self.valid_size
        )
        self.assertEqual(train_imgs, self.train_imgs)
        self.assertEqual(val_imgs, self.valid_imgs)
        self.assertEqual(train_lbls, self.train_lbls)
        self.assertEqual(val_lbls, self.valid_lbls)


if __name__ == '__main__':
    unittest.main()