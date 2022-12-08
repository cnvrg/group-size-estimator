# Group Size Estimator (GSE) Inference
## _cnvrg_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The GSE Inference library enables the user to run inference on an image or video by setting up an endpoint. Curl commands or Cnvrg's `Try it Live` feature can be used to make API calls to the endpoint. This library can be used as part of the [GSE Training Blueprints]() as well as the standalone [GSE Inference Blueprints]().

Click [here]() for more information on this library.

## Library Flow
The following list outlines this library's high-level flow:
- The user uploads a test image or video from their local file system.
- The model on the backend makes a prediction and displays the detected objects as well as object counts to the user.

## Inputs
The user can make calls to the API by using the curl command or python integration snippet on the endpoint page.

## Output
The sample output for an image looks as follows:
```bash
{
  "output": [
    {
      "bbox": [
        112,
        118,
        178,
        231
      ],
      "class": "person",
      "conf": 0.83
    },
    {
      "object": "person",
      "object_count": 1
    }
  ]
}
```

The sample output for a video looks as follows:
```bash
{
  "output": [
    {
      "test_frame1.jpg": [
        {
          "bbox": [
            527,
            433,
            1019,
            678
          ],
          "class": "person",
          "conf": 0.65
        },
        {
          "object": "person",
          "object_count": 1
        }
      ],
      "test_frame2.jpg": [
        {
          "bbox": [
            530,
            460,
            1100,
            678
          ],
          "class": "person",
          "conf": 0.7
        },
        {
          "object": "person",
          "object_count": 1
        }
      ]
    }
  ]
}
```