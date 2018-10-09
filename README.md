# Train a Wall-E
This repository provides a custom object-detection model to sort recyclabes from trash. 
 
## Motivation for this project:
USA produces approximately 254 million tons of trash per year. Almost 51% of its recyclable portion ends in land-fill. Further, the revenue from recyclables is expected to reach $ 435 billion in 2023 from $260 billion today. Hence there exists a huge market for quick and efficient sorting of recyclable items from trash.



## Requisites
- The dataset was obtained from [scrapsort repo] (https://github.com/patil215/scrapsort/tree/master/training_data/v2) trashnet. Images did not have bounding boxes and so they were labelled using labelImg. 
 

## Run Inference
- Run the predict.py file in keras-yolo2 folder with a directory of images as its input. The output of the inference i.e. predicted bounding boxes will be overlaid on top of the images and saved in the same directory.
```
# python predict.py -c config.json -w full_yolo_trash.h5 -i ~/data/processed/test/

```
