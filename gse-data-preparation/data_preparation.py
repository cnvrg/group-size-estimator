import argparse
import os
import pandas as pd
import shutil
import yaml
from sklearn.model_selection import train_test_split

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

# Read config file
with open(os.path.dirname(os.path.abspath(__file__)) + "/data_preparation_config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


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


class NumberOfClassesError(Exception):
    """Raise if number of classes is less than 2"""
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.num_classes = num_classes

    def __str__(self):
        return f"NumberOfClassesError: Number of classes is {self.num_classes}. You need to define atleast 1 class!"


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


def validate_dataset(data_dir_list, img_dir_list, lbl_dir_list, img_formats, class_list):
    """Validates the input dataset
    
    Checks if number of images is same as number of labels.
    Raises an error if an image does not have a corresponding label/annotation file.

    Args:
        data_dir_list:  list containing names of all files present in the data directory
        img_dir_list: list containing names of all files present in the image directory
        lbl_dir_list: list containing names of all files present in the label directory
        img_formats: A list containing image file formats compatible with YOLOv5
        class_list: A list containing all classes/categories in the dataset
    
    Raises:
        DatasetSizeError: If number of images is not equal to number of label/annotation files
        DatasetNamingError: If an image does not have a label/annotation file associated with it
        NumberOfClassesError: If number of classes is less than 1
    
    Returns:
        image_list: a list containing filenames for images
        label_list: a list containing filenames for labels/annotations
    """
    image_list, label_list = [], []

    if len(data_dir_list):
        image_list = [file for file in data_dir_list if any(fmt in file for fmt in img_formats)]
        label_list = [file for file in data_dir_list if ".txt" in file]
    else:
        image_list = [file for file in img_dir_list if any(fmt in file for fmt in img_formats)]
        label_list = [file for file in lbl_dir_list if ".txt" in file]

    if len(image_list) != len(label_list):
        raise DatasetSizeError(len(image_list), len(label_list))

    for image in image_list:
        image_name = image.split(".")[0]
        label_file = image_name + ".txt"
        if label_file not in label_list:
            raise DatasetNamingError(image)
        
    if len(class_list) < 1:
        raise NumberOfClassesError(len(class_list))
    
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


def prepare_dataset(data_directory, img_directory, lbl_directory, output_dir, train_images, val_images, train_labels, val_labels):
    """Creates training and validation directories and moves these to the output directory"""
    train_img_dst = os.path.join(output_dir, config['training_images_dir'])
    train_lbl_dst = os.path.join(output_dir, config['training_labels_dir'])
    valid_img_dst = os.path.join(output_dir, config['validation_images_dir'])
    valid_lbl_dst = os.path.join(output_dir, config['validation_labels_dir'])
    os.makedirs(train_img_dst, exist_ok=True)
    os.makedirs(train_lbl_dst, exist_ok=True)
    os.makedirs(valid_img_dst, exist_ok=True)
    os.makedirs(valid_lbl_dst, exist_ok=True)

    img_src, lbl_src = img_directory, lbl_directory
    if data_directory.lower() != "none":
        img_src, lbl_src = data_directory, data_directory

    for i in range(len(train_images)):
        shutil.copy(os.path.join(img_src, train_images[i]), train_img_dst)
        shutil.copy(os.path.join(lbl_src, train_labels[i]), train_lbl_dst)

    for i in range(len(val_images)):
        shutil.copy(os.path.join(img_src, val_images[i]), valid_img_dst)
        shutil.copy(os.path.join(lbl_src, val_labels[i]), valid_lbl_dst)


def create_dataset_config(class_list, output_dir):
    """Creates a dataset configuration (.yaml) file in the output directory"""
    yaml_file = open(os.path.join(output_dir, config['dataset_yaml_file']), 'w+')

    # Define keys for paths to training and validation sets
    yaml_file.write(config['train_key'] + ": " + os.path.join(output_dir, config['training_images_dir']) + "/" + "\n")
    yaml_file.write(config['valid_key'] + ": " + os.path.join(output_dir, config['validation_images_dir']) + "/" + "\n")

    # Define keys for number of classes and class names
    yaml_file.write(config['num_classes'] + ": " + str(len(class_list)) + "\n")
    yaml_file.write(config['class_names'] + ": " + str(class_list))
    yaml_file.close()


def data_preparation_main():
    """Command line execution."""
    # Get input parameters and perform validation
    args = parse_parameters()
    validate_arguments(args.data_dir, args.image_dir, args.label_dir, args.class_file, args.valid_size)
    image_formats = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]

    # Create lists for images, labels and classes
    data_dir_contents, img_dir_contents, lbl_dir_contents = [], [], []
    if args.data_dir.lower() != "none":
        data_dir_contents = os.listdir(args.data_dir)
    else:
        img_dir_contents = os.listdir(args.image_dir)
        lbl_dir_contents = os.listdir(args.label_dir)
    
    class_list = []
    if ".csv" in args.class_file:
        class_list = list(pd.read_csv(args.class_file)['classes'])
    else:
        class_file = open(args.class_file)
        class_list = class_file.readlines()
        class_list = [class_name.rstrip('\n') for class_name in class_list]
        class_file.close()

    images, labels = validate_dataset(data_dir_contents, img_dir_contents, lbl_dir_contents, image_formats, class_list)

    # Split dataset into training and validation sets
    train_images, valid_images, train_labels, valid_labels = train_valid_split(images, labels, float(args.valid_size))

    # Move training and validation datasets to output directory
    prepare_dataset(args.data_dir, args.image_dir, args.label_dir, args.output_dir, train_images, valid_images, train_labels, valid_labels)

    # Create a dataset configuration (.yaml) file in output directory
    create_dataset_config(class_list, args.output_dir)


if __name__ == "__main__":
    data_preparation_main()
    