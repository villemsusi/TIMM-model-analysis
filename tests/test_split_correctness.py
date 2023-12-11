import os

direc = "../oxford-iiit-pet/images"

for file in os.listdir(direc+"/test/"):
    for file2 in os.listdir(direc + "/train/"):
        if file==file2:
            print("ERROR!")
            break
    else:
        continue
    break
print("NO ERROR")     