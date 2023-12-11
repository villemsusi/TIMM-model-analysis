# Analysis of image classification models for use on low-resource computing units
### Villem Susi
### The goal of this project is to choose the best image classifier to work on a low-resource device, such as a Raspberry Pi. To achieve this goal, the aim is to analyse different pretrained PyTorch image classifiers from [timm](https://github.com/huggingface/pytorch-image-models), all of which will be improved upon via transfer learning on the [Oxford III-t Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).
## Project structure
* The dataset should live at /oxford-iiit-pet/
* /PetClassifier/ will be created upon training to hold the created model checkpoints
* /predictions/ will be created upon testing to hold data about predictions
* /pi_predictions/ holds sample inference data, created by running the inference script on a Raspberry Pi 4 (4GB) using 29 different models on a subset of the test images

## Setup
* Download and unpack images from [Oxford III-t Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- Unpack```tar -xf oxford-iiit-pet/images.tar -C oxford-iiit-pet/```
* Create Python Virtual Environment ```python -m venv <name>```
* Install necessary dependencies ```pip install -r requirements.txt```
* Create Train-Validation-Test Split ```python create_train_test_split.py```

## Training
* Setup Conda environment
* Install CUDA and cudNN
* Setup CUDA on Conda environment
* ```python scripts/train.py``` 
* Trained model checkpoint will be saved in /PetClassifier/
### CLI parameters
* --epochs -> <b>Required.</b> Specify the number of epochs to train
* --model -> <b>Required.</b> Specify which timm model to use for transfer learning, or <i>all</i> to train a selection of 29 models


## Inference
```python scripts/inference.py```
### CLI parameters
* --checkpoint -> <b>Required.</b> Specify proper path (absolute/relative from source directory) to the checkpoint file (.pth)
* --imgpath -> <b>Requires --label to be specified.</b> Specify a path to a single image to run inference on
* --label -> <b>Requires --imgpath to be specified.</b> Specify the label for the given inference image
* --dir -> Specify a directory of images to run inference on. This directory needs to be formatted in the following way:
    - directory
        - class1
            - class1_img1
            - class1_img2
        - class2
            - class2_img1
            - class2_img2

## Analysis
```python scripts/analyse.py```
### CLI parameters
* --rawdata -> <b>Requires --modeldata.</b> Specify the path to a raw data .csv file. 
* --modeldata -> <b>Requires --rawdata.</b> Specify the path to the model data .csv file.
These files need to be created by the inference.py script to have the required formatting. If these parameters are unspecified, the sample data files will be used.