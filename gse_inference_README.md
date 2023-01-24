The Group Size Estimator (GSE) Inference Blueprint provides an API endpoint for performing object counting on unseen images. This blueprint uses a pretrained YOLOv5 Small (yolov5s) model and allows the user to upload an image file to the endpoint. Complete the following steps to run this blueprint:

1. Click the **Use Blueprint** button.
2. Select a suitable Compute Template to run the inference endpoint and click the **Start** button.
3. Once the inference endpoint is ready, use either the Try it Live feature or the Integration panel to integrate the endpoint into your code.

Note: This blueprint serves as an example of the inference endpoint and cannot be used to make predictions on custom data. That being said, the YOLOv5 Small model used here has been pre-trained on the [COCO](https://cocodataset.org/#home) dataset and can be used to detect/count the 80 classes included in this dataset. Use this inference blueprint's [training]() counterpart to train models on your own custom data and establish an endpoint based on the newly trained model.