import argparse
import os
import shutil
import yaml

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


class ModelNotFoundError(Exception):
    """Raise if the model file cannot be found"""

    def __init__(self, model_path):
        super().__init__(model_path)
        self.model_path = model_path

    def __str__(self):
        return f"ModelNotFoundError: The model file does not exist at {self.model_path}. Please check the previous library!"


class ConfidenceValueError(Exception):
    """Raise if confidence value is not between 0 and 1"""

    def __init__(self, confidence):
        super().__init__(confidence)
        self.confidence = confidence

    def __str__(self):
        return f"ConfidenceValueError: The confidence value cannot be {self.confidence}. It needs to be between 0 and 1!"


def parse_parameters():  # pragma: no cover
    """Command line parser"""
    parser = argparse.ArgumentParser(description="""Batch Predict""")
    parser.add_argument(
        "--test_dir",
        action="store",
        dest="test_dir",
        required=True,
        help="""--- Path to directory containing test images ---""",
    )
    parser.add_argument(
        "--img_size",
        action="store",
        dest="img_size",
        required=False,
        default=640,
        help="""--- Size of images for batch prediction. Images will be resized to this size ---""",
    )
    parser.add_argument(
        "--confidence",
        action="store",
        dest="confidence",
        required=False,
        default=0.4,
        help=""" The confidence value for making final predictions ---""",
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


def validate_model_location(model_loc):
    """Validates the path to the trained model (best.pt)

    Checks if the trained model file exists at the specified location

    Args:
        model_loc: path to the trained model file in PyTorch format

    Raises:
        ModelNotFoundError: If trained model file cannot be found
    """
    if not os.path.exists(model_loc):
        raise ModelNotFoundError(model_loc)


def validate_confidence_value(confidence):
    """Checks if confidence value is between 0 and 1

    Args:
        confidence: confidence value for making final predictions

    Raises:
        ConfidenceValueError: If confidence value is not between 0 and 1
    """
    if float(confidence) <= 0 or float(confidence) >= 1:
        raise ConfidenceValueError(confidence)


def move_model_file(config_dict):  # pragma: no cover
    """Moves the model file to the working directory"""
    shutil.move(config_dict["model_loc"], config_dict["move_dest"])


def batchpredict_main():  # pragma: no cover
    """Command line execution"""
    # Get input parameters
    args = parse_parameters()

    # Read config file and perfrom validation
    with open(
        os.path.dirname(os.path.abspath(__file__)) + "/batchpredict_config.yaml", "r"
    ) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    validate_model_location(config_dict["model_loc"])
    validate_confidence_value(args.confidence)

    # Move model file to current working directory
    move_model_file(config_dict)

    # Run YOLOv5 detection script
    model_loc = config_dict["model_loc_new"]
    project_loc = args.output_dir + config_dict["project_name"]
    command = f"python detect.py --weights {model_loc} --img {args.img_size} --conf {args.confidence} --source {args.test_dir} --project {project_loc} --save-txt --save-conf --hide-conf"
    os.system(command)


if __name__ == "__main__":
    batchpredict_main()
