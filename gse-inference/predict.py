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

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

import base64
import cv2
import magic
import numpy as np
import os
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

# Specify acceptable file formats
img_formats = ["jpg", "jpeg", "png"]
vid_formats = ["mov", "avi", "mp4", "mpg", "mpeg", "m4v", "wmv", "mkv"]


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

    # Perform base 64 conversion on uploaded data
    decoded = base64.b64decode(data["media"][0])

    # Get file extension and define save path for test image/video
    file_ext = magic.from_buffer(decoded, mime=True).split("/")[-1]
    savepath = config["test_file_name"] + f".{file_ext}"

    # Process images and videos depending on file format
    if file_ext in img_formats:
        nparr = np.fromstring(decoded, np.uint8)
        test_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(savepath, test_img)

        # Make predictions
        result = run(weights=model_path, source=savepath, save_conf=True, save_txt=True)

        # create json response for image
        obj_info = result[savepath][0]
        obj_counts = result[savepath][1]
        for classname in obj_info:
            response["output"] += obj_info[classname]
            count_dict = {
                "object": classname,
                "object_count": obj_counts[classname],
            }
            response["output"].append(count_dict)

    elif file_ext in vid_formats:
        fh = open(savepath, "wb")
        fh.write(decoded)
        fh.close()

        # Make predictions
        result = run(weights=model_path, source=savepath, save_conf=True, save_txt=True)

        # create json response for video
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
