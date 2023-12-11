# IDS_project

## Analysis of image classification models for use on low-resource computing units

## Setup
* Download and unpack images from [https://www.robots.ox.ac.uk/~vgg/data/pets/, Oxford III-t Pet Dataset] ```tar -xf oxford-iiit-pet/images.tar -C oxford-iiit-pet/```
* Create Python Virtual Environment ```python -m venv <name>```
* Install necessary dependencies ```pip install -r requirements.txt```
* Create Train-Validation-Test Split ```python create_train_test_split.py```

## Training - !INCOMPLETE!
* Setup Conda environment
* Install CUDA and cudNN
* Setup CUDA on Conda environment
* Run train script ```python scripts/train.py``` -> checkpoint will be saved in PetClassifier directory
### train.py CLI parameters
* --epochs -> <b>Required.</b> Specify the number of epochs to train
* --model -> <b>Required.</b> Specify which timm model to use for transfer learning, or <i>all</i> to train a selection of 29 models


## Inference
* Run test script ```python scripts/inference.py```
### inference.py CLI parameters
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