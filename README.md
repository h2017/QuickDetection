# Train a Wall-E
This repository provides a custom object-detection model to sort recyclabes from trash. This [Google Slides](https://docs.google.com/presentation/d/1zeNFiWiGkVjS7e59hS5mzyFpvdS8Ayi5G3jqLYzYhPY/edit?usp=sharing) provides a briew overview of the project.
 
## Dataset
The dataset was obtained from <https://github.com/patil215/scrapsort/tree/master/training_data/v2>. It has 2400 images of common recyclable materials belonging to 5 classes - Metal , Paper, Cardboard, Glass, Plastic. The images contained trash objects at different angles and on a white background. I dropped the Paper class, as it was found to be problematic.  Images did not have bounding boxes and so I manually labelled approximately 170 images per class using [labelImg](https://github.com/tzutalin/labelImg). The annotated training data is available in data folder. 

## Model & Training
I have used keras-yolo v2 model. I modified config file to include my own anchor boxes and labels. Also, I have modified the loss function from the original, to suit my dataset. Before training the model, download [these weights](<https://code.et.stanford.edu/newmans/CS230/blob/b8c3aa0a181767adb495465f7e367e99b341778f/keras-yolo2/full_yolo_backend.h5>) to build the backbone. To train, do:

``` 
cd keras-yolo2
python train.py -c config.json
```

## Results
- Mean Average Precision = 0.84
- Confusion Matrix :
<p align="center">
 <img src="https://github.com/h2017/QuickDetection/blob/dev-data_pipeline-Sep_17_2018/ConfusionMatrix.png" width="400"> </p>


## Run Inference
Run the predict.py file in keras-yolo2 folder with a directory of images as its input. The output of the inference i.e. predicted bounding boxes, will be overlaid on top of the images and saved in the same directory.
```
cd keras-yolo2
python predict.py -c config.json -w full_yolo_trash.h5 -i ~/data/processed/test/
```
- Inference time (on Nvidia Tesla K80) = 0.061 +/- 0.09  (s)
- A sample test image: 
<p align="center">
 <img src="https://github.com/h2017/QuickDetection/blob/dev-data_pipeline-Sep_17_2018/SampleInferenceOutput.png"> </p>

## Transform keras model for serving
For model serving, it is better to reduce model loading time and computation time. I achieved that using the tensorflow graph transformation tool - [graph_transforms](<https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms>). The output of the tool is a frozen tensorflow graph. 

To obtained a frozen graph, do:
```
cd keras-yolo2
python keras_to_tensorflow.py -c config.json -w full_yolo_trash.h5
```
Often, you only know the name of the output layer and not the exact node name in that output layer. Then, you can visually find the output node in a network using tensorboard. For this, do:
```
cd keras-yolo2
python keras_to_tensorflow.py -c config.json -w full_yolo_trash.h5 -v True
```
An events file will be generated in the '/tmp/' folder. Then, run the following command in terminal and follow its output to visualize
```
tensorboard --logdir=/tmp/
```
