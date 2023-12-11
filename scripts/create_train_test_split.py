import os
import random
import shutil

random.seed(10)

## Moves files based on a set distribution to train-val-test splits
def splitRandom(train_size, val_size):
    for f in os.listdir(direc):
        file = os.path.join(direc,f)
        if not os.path.isfile(file):
            continue
        if not os.path.exists(direc+"train/"):
            os.mkdir(direc+"train/")
        if not os.path.exists(direc+"val/"):
            os.mkdir(direc+"val/")
        if not os.path.exists(direc+"test/"):
            os.mkdir(direc+"test/")
        rand = random.random()

        if rand <= train_size:
            shutil.move(file, direc+"train/"+f)
        elif rand <= train_size+val_size:
            shutil.move(file, direc+"val/"+f)
        else:
            shutil.move(file, direc+"test/"+f)

## Creates labelled subdirectories and moves every image to a correctly labelled subdirectory
def createSubdirectories(d):

    for i, f in enumerate(os.listdir(d)):
        f_abs_path = os.path.join(d, f)
        if not os.path.isfile(f_abs_path):
            continue
        if ".mat" in f:
            os.remove(f_abs_path)
        label = f.rpartition("_")[0]
        if label.startswith("train"):
            label = label.replace("train", "")
        elif label.startswith("validation"):
            label = label.replace("validation", "")
        elif label.startswith("test"):
            label = label.replace("test", "")
        if not os.path.exists(d+label):
            os.mkdir(d+label)

        shutil.move(f_abs_path, f"{d}{label}/{label}_{i}.jpg")


if __name__ == "__main__":

    direc = "../oxford-iiit-pet/images/"

    splitRandom(0.75, 0.15, 0.1)

    for subdirec in ["train/", "val/", "test/"]:
        createSubdirectories(direc+subdirec)