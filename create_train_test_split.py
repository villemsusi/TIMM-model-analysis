import os
import random
import shutil

random.seed(10)

direc = "oxford-iiit-pet/images/"


def splitRandom(train_size, val_size, test_size):
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

def createSubdirectories():
    for subdirec in ["train/", "val/", "test/"]:
        for i, f in enumerate(os.listdir(direc+subdirec)):
            f_abs_path = os.path.join(direc, subdirec, f)
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
            if not os.path.exists(direc+subdirec+label):
                os.mkdir(direc+subdirec+label)

            shutil.move(f_abs_path, f"{direc}{subdirec}{label}/{label}_{i}.jpg")
        else:
            continue
        break

if __name__ == "__main__":
    splitRandom(0.75, 0.15, 0.1)

    createSubdirectories()