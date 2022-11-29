# Group Size Estimator (GSE) Finetuning
## _cnvrg_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The GSE Finetuning library lets the user finetune a pre-trained [YOLOv5](https://github.com/ultralytics/yolov5) model on custom data. End-users can pass hyperparameters like initial model weights, number of epochs, batch size etc. to finetune the model. As part of the [GSE Training Blueprints](), this library reads the dataset prepared by the previous library, use it to finetune the model and then saves the model for running inference and batch prediction.

Click [here]() for more information on this library.

## Library Flow
The following list outline the library's high-level flow:
- The user configures a set of hyperparameters to finetune the YOLOv5 model.
- The library then calls the YOLOv5 training script to perform finetuning.
- The library also creates a project directory in the output directory to store the trained model, metrics, plots and results.

## Inputs
This library relies on the dataset directories and config file created by the Data Preparation library. The expected directory structure is as follows.
```
| - basedir
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

The GSE Finetuning library requires the following inputs:
* `--model_weights` - string, optional. Provide the YOLOv5 variant that the user wants to finetune. Some common variants are YOLOv5 Small (yolov5s.pt) and YOLOv5 Medium (yolov5m.pt). Default value: `yolov5s.pt`.
* `--img_size` - string, optional. Provide the size of input images for finetuning the model. The images will be resized if the original size is different from the size provided. Default value: `640`.
* `--batch_size` - string, optional. Specify the batch size to train the model with. Default value: `16`.
* `--num_epochs` - string, optional. Number of epochs to train the model for. Default value: `5`.

Note: Most of the YOLOv5 variants (YOLOv5s, YOLOv5m, etc.) have been pre-trained using the [COCO](https://cocodataset.org/#home) dataset.

## Sample Command
Refer to the following sample command:

```bash
python finetune.py --model_weights yolov5s.pt --img_size 640 --batch_size 16 --num_epochs 5
```

## Outputs
The GSE Finetuning library generates the following outputs:
- The library generates the following directory structure.
```
| - outputdir
    | - runs
        | - train
            | - exp
                | - weights
                    | - best.pt
                    | - last.pt
                | - metrics.png
                | - plot.jpg
                | - results.csv
                | ..
```
- The library writes all files created to the default path `/cnvrg`.
- All these files can be used by subsequent libraries in the Blueprint.

## Troubleshooting
- Ensure the input arguments to the library are valid and accurate.
- Check the experiment's Artifacts section to confirm the library has generated the output files/directories.