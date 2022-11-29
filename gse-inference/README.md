# Group Size Estimator (GSE) Inference
## _cnvrg_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The GSE Inference library enables the user to run inference on an image by setting up an endpoint. Curl commands or Cnvrg's `Try it Live` feature can be used to make API calls to the endpoint. This library can be used as part of the [GSE Training Blueprints]() as well as the standalone [GSE Inference Blueprints]().

Click [here]() for more information on this library.

## Library Flow
The following list outlines this library's high-level flow:
- The user uploads a test image from his local file system.
- The model on the backend makes a prediction and displays the detected objects as well as object counts to the user.

## Inputs
The user can make calls to the API using the following example curl commands:
* The user can pass the path to a test image as input to the endpoint.
```bash
curl -X POST \
    {link to your deployed endpoint} \
-H 'Cnvrg-Api-Key: {your_api_key}' \
-H 'Content-Type: application/json' \
-d '{"vars": ""}'
```

## Output
The sample output looks as follows:
```bash
{"output": ""}
```