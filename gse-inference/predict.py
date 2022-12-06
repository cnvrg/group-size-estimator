import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

import base64
import cv2
import magic
import numpy as np
import os
import shutil
import yaml
from detect import run

# Define path to gse-inference library for dev or prod
lib_path = "/cnvrg_libraries/dev-gse-inference/"
if os.path.exists("/cnvrg_libraries/gse-inference"):
    lib_path = "/cnvrg_libraries/gse-inference/"

# Read config file
with open(lib_path + "inference_config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


def find_model(trained_path, standalone_path):
    """Returns the path to the trained model file"""
    if os.path.exists(trained_path):
        return trained_path
    return standalone_path


# Get location of object counter model
model_path = find_model(
    config["trained_model_path"], lib_path + config["standalone_model_name"]
)


def predict(data):
    """Performs object counting on an image provided as input to the webservice
    Args:
        data: a json object representing input data
    Returns:
        response: dictionary containing the prediction/results
    """
    # Define json response
    response = {}
    response["output"] = []

    # Create directory to store images and define  counter
    img_dir = os.path.join(os.getcwd(), config["test_img_dir"])
    os.mkdir(img_dir)
    counter = 0

    for image in data["img"]:
        # Perform base 64 conversion on uploaded data
        decoded_img = base64.b64decode(image)
        counter += 1

        # Convert buffer to numpy array and save uploaded image
        file_ext = magic.from_buffer(decoded_img, mime=True).split("/")[-1]
        nparr = np.fromstring(decoded_img, np.uint8)
        test_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        savepath = os.path.join(
            img_dir, config["test_file_prefix"] + f"{counter}.{file_ext}"
        )
        cv2.imwrite(savepath, test_img)

    # Make predictions
    result = run(weights=model_path, source=img_dir, save_conf=True, save_txt=True)
    shutil.rmtree(img_dir)

    # Create JSON response for image
    for filename in result:
        file_dict = {}
        file_dict[filename] = []
        obj_info = result[filename][0]
        obj_counts = result[filename][1]
        for classname in obj_info:
            file_dict[filename] += obj_info[classname]
            count_dict = {
                "object": classname,
                "object_count": obj_counts[classname],
            }
            file_dict[filename].append(count_dict)
        response["output"].append(file_dict)

    return response
