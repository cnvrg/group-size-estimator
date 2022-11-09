import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


class NoneDatasetError(Exception):
    """Raise if data directory, images directory and labels directory are all None"""

    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return "NoneDatasetError: No dataset provided. The first three input arguments cannot be None!"


class DatasetPathError(Exception):
    """Raise if either the images directory or labels directory is None"""
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return "DatasetPathError: The images directory or label directory cannot be None!"


class ClassFileFormatError(Exception):
    """Raise if the classes file is not in txt or csv format"""
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return "ClassFileFormatError: The classes file needs to be in txt or csv format!"


class ValidationSizeError(Exception):
    """Raise if validation size is not between 0.0 and 0.4"""

    def __init__(self, valid_size):
        super().__init__(valid_size)
        self.valid_size = valid_size

    def __str__(self):
        return f"ValidationSizeError: {self.valid_size} is an invalid validation size. Validation size needs to be a value between 0.0 and 0.4!"


class DatasetSizeError(Exception):
    """Raise if number of images is not equal to number of labels/annotations"""
    def __init__(self, num_images, num_labels):
        super().__init__(num_images, num_labels)
        self.num_images = num_images
        self.num_labels = num_labels
    
    def __str__(self):
        return f"DatasetSizeError: The number of images should be equal to number of labels/annotations. Your dataset contains {self.num_images} images and {self.num_labels} labels"


class DatasetNamingError(Exception):
    """Raise if an image does not have a corresponding label"""
    def __init__(self, image):
        super().__init__(image)
        self.image = image
    
    def __str__(self):
        return f"DatasetNamingError: No label found for image {self.image}. Images and labels should have the same file names without extensions!"


def parse_parameters():
    """Command line parser."""
    parser = argparse.ArgumentParser(description="""Dataset Preparation""")
    parser.add_argument("--data_dir", action="store", dest="data_dir", required=True, help="""--- Path to directory containing both images and labels ---""")
    parser.add_argument("--image_dir", action="store", dest="image_dir", required=True, help="""--- Path to directory containing images ---""")
    parser.add_argument("--label_dir", action="store", dest="label_dir", required=True, help="""--- Path to directory containing labels/annotations ---""")
    parser.add_argument("--class_file", action="store", dest="class_file", required=True, help="""--- Path to .txt or .csv file containing class names ---""")
    parser.add_argument("--output_dir", action="store", dest="output_dir", required=False, default=cnvrg_workdir, help="""--- The path to save library artifacts to ---""")
    parser.add_argument("--valid_size", action="store", dest="valid_size", required=False, default=0.1, help="""--- Size of validation set as percentage of entire dataset ---""")
    return parser.parse_args()


def validate_arguments(data_directory, img_directory, lbl_directory, cls_file, valid_size):
    """Validates input arguments
    
    Checks if the input arguments provided by the user are appropriate for running this library.
    Makes sure that the first three arguments are not None and that the class file is in .txt or .csv format
    
    Args:
        data_directory: The directory containing both images as well as labels
        img_directory: The directory containing just images
        lbl_directory: The directory containing just labels/annotations
        cls_file: The file containing pre-defined classes/categories for object detection
        valid_size: Size of validation set as percentage of entire dataset
    
    Raises:
        NoneDatasetError: If first three input arguments are all None
        DatasetPathError: If either images directory or labels directory is None
        ClassFileFormatError: If the classes file is not in txt or csv format
        ValidationSizeError: If validation size is not between 0.0 and 0.4
    """
    if data_directory.lower() == "none" and img_directory.lower() == "none" and lbl_directory.lower() == "none":
        raise NoneDatasetError

    if data_directory.lower() == "none":
        if (img_directory.lower() != "none" and lbl_directory.lower() == "none") or (img_directory.lower() == "none" and lbl_directory.lower() != "none"):
            raise DatasetPathError
    
    if not any(ext in cls_file for ext in [".txt", ".csv"]):
        raise ClassFileFormatError

    if not (float(valid_size) > 0 and float(valid_size) <= 0.4):
        raise ValidationSizeError(float(valid_size))    


def validate_dataset(data_directory, img_directory, lbl_directory, img_formats):
    """Validates the input dataset
    
    Checks if number of images is same as number of labels.
    Raises an error if an image does not have a corresponding label/annotation file.

    Args:
        data_directory: The directory containing both images as well as labels
        img_directory: The directory containing just images
        lbl_directory: The directory containing just labels/annotations
        img_formats: A list containing image file formats compatible with YOLOv5
    
    Raises:
        DatasetSizeError: If number of images is not equal to number of label/annotation files
        DatasetNamingError: If an image does not have a label/annotation file associated with it
    
    Returns:
        image_list: a list containing filenames for images
        label_list: a list containing filenames for labels/annotations 
    """
    image_list, label_list = [], []

    if data_directory.lower() != "none":
        file_list = os.listdir(data_directory)
        image_list = [file for file in file_list if any(fmt in file for fmt in img_formats)]
        label_list = [file for file in file_list if ".txt" in file]
    else:
        image_files = os.listdir(img_directory)
        label_files = os.listdir(lbl_directory)
        image_list = [file for file in image_files if any(fmt in file for fmt in img_formats)]
        label_list = [file for file in label_files if ".txt" in file]

    if len(image_list) != len(label_list):
        raise DatasetSizeError(len(image_list), len(label_list))

    for image in image_list:
        image_name = image.split(".")[0]
        label_file = image_name + ".txt"
        if label_file not in label_list:
            raise DatasetNamingError(image)
    
    return image_list, label_list


def train_valid_split(images, labels, valid_size):
    """Splits data into training and validation sets
    
    Args:
        images: a list containing filenames for images
        labels: a list containing filenames for labels/annotations
        valid_size: size of validation set as percentage of entire dataset

    Returns:
        train_images: a list containing image filenames from the training set
        val_images: a list containing image filenames from the validation set
        train_labels: a list containing label/annotation filenames from the training set
        train_labels: a list containing label/annotation filenames from the validation set
    """
    images.sort()
    labels.sort()

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=valid_size, random_state=1)

    return train_images, val_images, train_labels, val_labels


def prepare_dataset(data_directory, img_directory, lbl_directory, train_images, val_images, train_labels, val_labels):
    """Creates training and validation directories and moves these to the working directory"""
    pass


def data_preparation_main():
    """Command line execution."""
    # Get input parameters and perform validation
    args = parse_parameters()
    validate_arguments(args.data_dir, args.image_dir, args.label_dir, args.class_file, args.valid_size)
    image_formats = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
    images, labels = validate_dataset(args.data_dir, args.image_dir, args.label_dir, image_formats)

    # Split dataset into training and validation sets
    train_images, valid_images, train_labels, valid_labels = train_valid_split(images, labels, args.valid_size)


if __name__ == "__main__":
    data_preparation_main()