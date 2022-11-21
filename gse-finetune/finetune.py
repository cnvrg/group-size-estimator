import argparse
import os
import shutil
import yaml

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


class DirectoryNotFoundError(Exception):
    """Raise if a directory cannot be found"""

    def __init__(self, dir_path):
        super().__init__(dir_path)
        self.dir_path = dir_path

    def __str__(self):
        return f"DirectoryNotFoundError: The required directory does not exist at {self.dir_path}. Please check the previous library!"


class ConfigNotFoundError(Exception):
    """Raise if the dataset config file does not exist"""

    def __init__(self, config_path):
        super().__init__(config_path)
        self.config_path = config_path

    def __str__(self):
        return f"ConfigNotFoundError: The dataset config file does not exist at {self.config_path}. Please check the previous library!"


def parse_parameters():  # pragma: no cover
    """Command line parser"""
    parser = argparse.ArgumentParser(description="""YOLOv5 Finetuning""")
    parser.add_argument(
        "--model_weights",
        action="store",
        dest="model_weights",
        required=False,
        default="yolov5s.pt",
        help="""--- Pre-trained model weights ---""",
    )
    parser.add_argument(
        "--img_size",
        action="store",
        dest="img_size",
        required=False,
        default=640,
        help="""--- Size of images for training and validation. Images will be resized to this size ---""",
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        dest="batch_size",
        required=False,
        default=16,
        help="""--- Batch size for training the model ---""",
    )
    parser.add_argument(
        "--num_epochs",
        action="store",
        dest="num_epochs",
        required=False,
        default=5,
        help="""--- Number of epochs to train the model for ---""",
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        dest="output_dir",
        required=False,
        default=cnvrg_workdir,
        help="""--- The path to save library artifacts to ---""",
    )
    return parser.parse_args()


def validate_file_locations(img_dir, lbl_dir, data_config):
    """Validates file paths

    Checks if data directories and the dataset config file are present at the specified locations

    Args:
        img_dir: path to the directory containing images
        lbl_dir: path to the directory containing labels
        data_config: path to the dataset config (.yaml) file

    Raises:
        DirectoryNotFoundError: If images or labels directroy cannot be found
        ConfigNotFoundError: If the dataset config file cannot be located
    """
    if not os.path.exists(img_dir):
        raise DirectoryNotFoundError(img_dir)

    if not os.path.exists(lbl_dir):
        raise DirectoryNotFoundError(lbl_dir)

    if not os.path.exists(data_config):
        raise ConfigNotFoundError(data_config)


def move_data_files(config_dict):  # pragma: no cover
    """Moves directories and the dataset config file to the working directory"""
    shutil.move(config_dict["images_loc"], config_dict["move_dest"])
    shutil.move(config_dict["labels_loc"], config_dict["move_dest"])
    shutil.move(config_dict["config_loc"], config_dict["move_dest"])


def finetune_main():  # pragma: no cover
    """Command line execution"""
    # Get input parameters
    args = parse_parameters()

    # Read config file and perform validation
    with open(
        os.path.dirname(os.path.abspath(__file__)) + "/finetune_config.yaml", "r"
    ) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    validate_file_locations(
        config_dict["images_loc"], config_dict["labels_loc"], config_dict["config_loc"]
    )

    # Move data directories and config file to current working directory
    move_data_files(config_dict)

    # Run YOLOv5 training script
    dataset_yaml_loc = config_dict["config_loc_new"]
    project_loc = args.output_dir + config_dict["project_name"]
    command = f"python train.py --weights {args.model_weights} --img {args.img_size} --batch {args.batch_size} --data {dataset_yaml_loc} --project {project_loc}"
    os.system(command)


if __name__ == "__main__":
    finetune_main()
