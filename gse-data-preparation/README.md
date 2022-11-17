# Group Size Estimator (GSE) Data Preparation
## _cnvrg_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The GSE Data Preparation library allows the user to create a dataset that is compatible with the YOLOv5 model. It accepts arguments such as paths to data/image/label directories, performs train-validation splitting and creates a dataset configuration file for training the model. As part of the [Group Size Estimator Blueprint](), this library processes raw data and makes it accessible to subsequent libraries in the Blueprint.

Click [here]() for more information on this library.

## Library Flow
The following list outlines this library's high-level flow:
- The user defines paths to the data/image/label directories and class file and selects an appropriate validation set size. 
- The library then reads data files (images and labels/annotations) from these paths, splits the data and creates directories for training and validation.
- The library also creates a dataset configuration file (.yaml) which can be used to train/finetune the model.

## Inputs
This library assumes that the user has access to the raw dataset via Connectors. The input dataset needs to consist of images (.jpeg, .png etc.) and annotations/labels (.txt) in YOLO format. More information on the YOLO format can be found [here](https://www.edge-ai-vision.com/2022/04/exploring-data-labeling-and-the-6-different-types-of-image-annotation/#:~:text=YOLO%3A%20In%20the%20YOLO%20labeling,coordinates%2C%20height%2C%20and%20width.).
The GSE Data Preparation library requires the following inputs:
* `--data_dir` - string, required. Provide the path to a directory containing both images and annotations/labels. The library will ignore the `--image_dir` and `--label_dir` if this argument is not `None`.
* `--image_dir` - string, required. Provide the path to a directory containing just images. To use this argument, `--data_dir` must be `None`.
* `--label_dir` - string, required. Provide the path to a directory containing just labels. To use this argument, `--data_dir` must be `None`.
* `--class_file` - string, required. Provide the path to the file containing all class/category names. This file needs to be in txt or csv format.
* `--valid_size` - string, optional. Provide the expected size of the validation set. Default value: `0.3`.

Note: Make sure that your data/image/label directories and class file exist in the same base directory. Here are some examples.
```
| - basedir
    | - images
        | - img1.jpg
        | - img2.jpg
        | ..
    | - labels
        | - img1.txt
        | - img2.txt
        | ..
    | - classes.txt
```
OR
```
| - basedir
    | - data
        | - img1.jpg
        | - img2.jpg
        | ..
        | - img1.txt
        | - img2.txt
        |..
    | - classes.csv
```

## Sample Command
Refer to the following sample command:

```bash
python data_preparation.py --data_dir None --image_dir /input/s3_connector/basedir/images  --label_dir /input/s3_connector/base_dir/labels --class_file /input/s3_connector/base_dir/classes.txt --valid_size 0.2
```

## Outputs
The GSE Preprocess library generates the following outputs:
- The library generates the following directory structure.
```
| - outputdir
    | - images
        | - train
            | - img1.jpg
            | - img20.jpg
            | ..
        | - val
            | - img10.jpg
            | - img102.jpg
            | ..
    | - labels
        | - train
            | - img1.txt
            | - img20.txt
            | ..
        | - val
            | - img10.txt
            | - img102.txt
            | ..
    | - dataset.yaml
```
- The library writes all files created to the default path `/cnvrg`.
- All these files can be used by subsequent libraries in the Blueprint.

## Troubleshooting
- Ensure the input arguments to the library are valid and accurate.
- Check the experiment's Artifacts section to confirm the library has generated the output files/directories.