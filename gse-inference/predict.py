import base64
import cv2
import os
import magic
import numpy as np
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
model_path = find_model(config["trained_model_path"], config["standalone_model_path"])


def predict(data):
    """Performs object counting on an image provided as input to the webservice
    Args:
        data: a json object representing input data
    Returns:
        response: dictionary containing the prediction/results
    """
    # Perfrom base 64 conversion on uploaded data
    decoded_img = base64.b64decode(data["img"])

    # Convert buffer to numpy array and save uploaded image
    file_ext = magic.from_buffer(decoded_img, mime=True).split("/")[-1]
    nparr = np.fromstring(decoded_img, np.uint8)
    test_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    savepath = config["test_file_name"] + f".{file_ext}"
    cv2.imwrite(savepath, test_img)

    # Make predictions
    result = run(weights=model_path, source=savepath, save_conf=True)
    os.remove(savepath)

    # Create JSON response
    response = {}
    response["output"] = []
    for filename in result:
        obj_info = result[filename][0]
        obj_counts = result[filename][1]
        for classname in obj_info:
            response["output"] += obj_info[classname]
            count_dict = {"object": classname, "object_count": obj_counts[classname]}
            response.append(count_dict)

    return response
