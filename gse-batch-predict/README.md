# Group Size Estimator (GSE) Batch Predict
## _cnvrg_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The GSE Batch Predict library enables the user to perform object counting on a set of test images. This library acccepts input arguments like path to the directory containing test images, the size of test images and the confidence value to consider while making final predictions. As part of the [GSE Training Blueprints](), it uses a trained model to make predictions and finally writes the results to the output directory.

Click [here]() for more information on this library.

## Library Flow
The following list outlines the library's high level flow:
- The user specifies the location of the directory containing test images and defines other parameters like image size and confidence value.
- The library then calls the YOLOv5 detection script to perform batch prediction
- The library also creates a project directory in the output directory to store images with predictions and other resulting artifacts.

## Inputs
This library assumes that the user has already finetuned a YOLOv5 model. This model will then be used to perform batch prediction. Connectors may be used to download the test dataset which should contain just images (.jpeg, .png etc.). 
The GSE Batch Predict library requires the following inputs:
* `--test_dir` - string, required. Provide the path to a directory containing test images.
* `--img_size` - string, optional. Provide the size of input images for performing batch prediction. The images will be resized if the original size is different from the size provided. Default value: `640`.
* `--confidence` - string, optional. Provide the confidence value (probability threshold) for making final predictions. Default value: `0.4`.

## Sample Command
Refer to the following sample command:

```bash
python batchpredict.py --test_dir /input/s3_connector/testdir/ --img_size 640 --confidence 0.3
```

## Outputs
The GSE Batch Predict library generates the following outputs:
- The library generates the following directory structure. Here, `imgx_counts.txt` contains the object counts, `imgx.txt` contains the predicted labels and `imgx.jpg` is the output image with bounding boxes and class labels.
```
| - outputdir
    | - runs
        | - detect
            | - exp
                | - counts
                    | - img1_counts.txt
                    | - img2_counts.tx
                    | ..
                | - labels
                    | - img1.txt
                    | - img2.txt
                    | ..
                | - img1.jpg
                | - img2.jpg
                | ..
```
- The library writes all files created to the default path `/cnvrg`.

## Troubleshooting
- Ensure the input arguments to the library are valid and accurate.
- Check the experiment's Artifacts section to confirm the library has generated the output files/directories.