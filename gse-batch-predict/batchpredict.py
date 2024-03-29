# Copyright (c) 2023 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

import argparse
import os
import yaml

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


class ModelNotFoundError(Exception):
    """Raise if the model file cannot be found"""

    def __init__(self, model_path):
        super().__init__(model_path)
        self.model_path = model_path

    def __str__(self):
        return f"ModelNotFoundError: The model file does not exist at {self.model_path}. Please check the previous library!"


class CommandFailedError(Exception):
    """Raise if os.system() returns non-zero exit code"""

    def __init__(self, exit_code):
        super().__init__(exit_code)
        self.exit_code = exit_code

    def __str__(self):
        return f"CommandFailedError: The os command returned exit code {self.exit_code}. Exit code cannot be non-zero!"


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

    # Run YOLOv5 detection script
    model_loc = config_dict["model_loc"]
    project_loc = args.output_dir + config_dict["project_name"]
    command = f"python detect.py --weights {model_loc} --source {args.test_dir} --project {project_loc} --save-txt --save-conf --hide-conf"
    exit_code = os.system(command)
    if exit_code != 0:
        raise CommandFailedError(exit_code)


if __name__ == "__main__":
    batchpredict_main()
